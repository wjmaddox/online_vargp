import math
import argparse
import pandas as pd
import time
import math
import torch

from botorch.sampling import SobolQMCNormalSampler
from botorch.optim.fit import fit_gpytorch_torch
from torch.distributions import Bernoulli
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood
from gpytorch.priors import GammaPrior

from volatilitygp.models import SingleTaskVariationalGP
from volatilitygp.likelihoods.binomial_likelihood import BinomialLikelihood


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return x.squeeze(-1)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--dataset", type=str, default="civ")
    parser.add_argument("--n_batch", type=int, default=100)
    parser.add_argument("--num_init", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--batch_limit", type=int, default=64)
    parser.add_argument("--inner_samples", type=int, default=16)
    parser.add_argument("--outer_samples", type=int, default=16)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--loss", type=str, default="elbo")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--ind_models", action="store_true")
    parser.add_argument("--eval_on_full_set", action="store_true")
    return parser.parse_args()


def entropy_via_threshold(f, threshold=0.1):
    bern_entropy = Bernoulli(logits=f).entropy()
    spiked_bern_entropy = bern_entropy * (f > math.log(threshold / (1 - threshold)))
    return spiked_bern_entropy.mean(0).sum(-1)


def entropy_reduction(model, batch_set, test_set, inner_samples=32, outer_samples=16):
    inner_sampler = SobolQMCNormalSampler(inner_samples)
    outer_sampler = SobolQMCNormalSampler(outer_samples)

    original_entropy = entropy_via_threshold(inner_sampler(model.posterior(test_set)))

    fantasy_model = model.fantasize(
        batch_set, sampler=inner_sampler, observation_noise=True
    )
    fantasy_entropy = entropy_via_threshold(
        outer_sampler(fantasy_model.posterior(test_set))
    )

    return (original_entropy - fantasy_entropy).clamp(min=0.0).sum(0)


def main(
    dataset: str = "civ",
    seed: int = 0,
    num_init: int = 100,
    batch_size: int = 1,
    n_batch: int = 100,
    inner_samples: int = 16,
    outer_samples: int = 16,
    batch_limit: int = 64,
    output: str = "results.pt",
    random: bool = False,
    beta: float = 0.1,
    loss: str = "elbo",
    lr: float = 0.01,
    eval_on_full_set: bool = False,
    recycle_lengthscales: bool = True,
):
    verbose = True

    data = pd.read_csv("data/" + dataset + "_data.csv")
    if dataset == "civ" or dataset == "hti":
        threshold = 0.1
    else:
        threshold = 0.02

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## split data

    full_x = torch.tensor(data.iloc[:, :-2].values).to(device)
    full_ground_truth_prob = torch.tensor(data.iloc[:, -1].values).to(device)

    perm = torch.randperm(data.shape[0])
    train_inds = perm[:num_init]
    test_inds = perm[num_init:]

    train_x = full_x[train_inds]
    train_ground_truth_prob = full_ground_truth_prob[train_inds]

    test_x = full_x[test_inds]
    test_ground_truth_prob = full_ground_truth_prob[test_inds]

    # draw samples from ground truth probability
    train_y = (
        torch.distributions.Binomial(total_count=100, probs=train_ground_truth_prob)
        .sample()
        .unsqueeze(-1)
    )
    test_y = (
        torch.distributions.Binomial(total_count=100, probs=test_ground_truth_prob)
        .sample()
        .unsqueeze(-1)
    )

    ## normalize data to [0, 1]^d
    mins = train_x.min(0)[0]
    maxes = train_x.max(0)[0]
    train_x = (train_x - mins) / (maxes - mins)
    test_x = (test_x - mins) / (maxes - mins)

    hotspot_acc_list, hotspot_mse_list, hotspot_sens_list = [], [], []
    hotspot_sampled_acc_list = []

    for iteration in range(n_batch):
        t0 = time.time()
        ## define model
        covar_module = ScaleKernel(
            MaternKernel(
                ard_num_dims=8,
                nu=1.5,
                lengthscale_prior=GammaPrior(3.0, 6.0),
                outputscale_prior=GammaPrior(2.0, 0.15),
            )
        )
        if iteration > 0 and recycle_lengthscales:
            print("recycling lengthscales")
            covar_module.outputscale = old_outputscale
            covar_module.base_kernel.lengthscale = old_lengthscale

        model = SingleTaskVariationalGP(
            likelihood=BinomialLikelihood(),
            init_points=train_x,
            init_targets=train_y.squeeze(-1),
            num_inducing=train_x.shape[0],
            use_piv_chol_init=True,
            learn_inducing_locations=True,
            covar_module=covar_module,
        )

        if loss == "elbo":
            mll = VariationalELBO(
                model.likelihood, model, num_data=train_x.shape[0], beta=beta
            )
        elif loss == "pll":
            mll = PredictiveLogLikelihood(
                model.likelihood, model, num_data=train_x.shape[0], beta=beta
            )

        fit_gpytorch_torch(mll, options={"lr": lr, "maxiter": 1000})

        ### record hotspot probability
        with torch.no_grad():
            model.eval()

            # for some ungodly reason andrade-pacheco et al evaluate on the full set, not the heldout set
            # thus we need to predict over all of the data (training data included)
            if eval_on_full_set:
                # we re-apply normalization
                x_for_pred = (full_x - mins) / (maxes - mins)
                gt_for_pred = full_ground_truth_prob
            else:
                x_for_pred = test_x
                gt_for_pred = test_ground_truth_prob

            pred_dist = model(x_for_pred)
            pred_prob = (pred_dist.mean.mul(-1).exp() + 1).reciprocal()

            true_is_hotspot = gt_for_pred > threshold

            # lets see if this is more accurate
            hotspot_samples = pred_dist.sample(torch.Size((512,)))
            hotspot_sampled_prob = (hotspot_samples.mul(-1).exp() + 1).reciprocal()
            hotspot_sampled_pred = (hotspot_sampled_prob > threshold).sum(0) > 256
            hotspot_sampled_acc = (
                ((hotspot_sampled_pred > threshold) == true_is_hotspot)
                .float()
                .mean()
                .cpu()
                .item()
            )

            hotspot_acc = (
                ((pred_prob > threshold) == true_is_hotspot).float().mean().cpu().item()
            )
            hotspot_mse = (pred_prob - gt_for_pred).pow(2).mean().cpu().item()

            hotspot_sens = (
                (pred_prob > threshold).float() * true_is_hotspot.float()
            ).sum().cpu().item() / true_is_hotspot.float().sum().cpu().item()

        if not random:
            ### now select a new point
            entropy_list = []
            for start in range(0, test_x.shape[0] + batch_limit, batch_limit):
                [p.detach_() for p in model.parameters()]
                # TODO: batch size of 10 via cyclic optimization
                query_points = test_x[start : (start + batch_limit)].unsqueeze(-2)
                if query_points.shape[0] > 0:
                    entropy = (
                        entropy_reduction(
                            model, query_points, test_x, inner_samples, outer_samples
                        )
                        .sum(-1)
                        .detach()
                        .cpu()
                    )
                    entropy_list.append(entropy)

            if batch_size == 1:
                best_point = torch.cat(entropy_list).argmax()
            else:
                raise NotImplementedError("oops, batch size of 1 is not implemented")
        else:
            # best point is randomly selected
            best_point = torch.randperm(test_x.shape[0])[:batch_size]
            if batch_size == 1:
                best_point = best_point.item()
            entropy_list = None

        train_x = torch.cat((train_x, test_x[best_point].unsqueeze(0)))
        train_y = torch.cat((train_y, test_y[best_point].unsqueeze(0)))
        train_ground_truth_prob = torch.cat(
            (train_ground_truth_prob, test_ground_truth_prob[best_point].unsqueeze(0))
        )

        test_x = torch.cat((test_x[:best_point], test_x[(best_point + 1) :]))
        test_y = torch.cat((test_y[:best_point], test_y[(best_point + 1) :]))
        test_ground_truth_prob = torch.cat(
            (
                test_ground_truth_prob[:best_point],
                test_ground_truth_prob[(best_point + 1) :],
            )
        )

        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: current_value (acc, sacc, mse, sens) = "
                f"({hotspot_acc:>4.2f},  {hotspot_sampled_acc:>4.2f}, {hotspot_mse:>4.2f} {hotspot_sens:>4.2f}, "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".")

        hotspot_acc_list.append(hotspot_acc)
        hotspot_sampled_acc_list.append(hotspot_sampled_acc)
        hotspot_mse_list.append(hotspot_mse)
        hotspot_sens_list.append(hotspot_sens)

        old_lengthscale = model.covar_module.base_kernel.lengthscale.detach()
        old_outputscale = model.covar_module.outputscale.detach()
        del model, entropy_list

        torch.cuda.empty_cache()
        print("memory allocated: ", torch.cuda.memory_allocated(device) / (1024 ** 3))

    output_dict = {
        "results": [
            torch.tensor(hotspot_acc_list),
            torch.tensor(hotspot_mse_list),
            torch.tensor(hotspot_sens_list),
            torch.tensor(hotspot_sampled_acc_list),
        ],
        "data": {"x": train_x, "theta": train_ground_truth_prob, "y": train_y},
    }
    return output_dict


if __name__ == "__main__":
    args = parse()
    args.recycle_lengthscales = not args.ind_models
    del args.ind_models
    output_dict = main(**vars(args))
    output_dict["pars"] = vars(args)

    torch.save(output_dict, args.output)
