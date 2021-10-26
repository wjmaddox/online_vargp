import argparse
import time
import torch
import gpytorch

from botorch.acquisition.active_learning import qNegIntegratedPosteriorVariance as qNIPV
from botorch.models import FixedNoiseGP
from botorch.optim.fit import fit_gpytorch_torch
from botorch.optim import optimize_acqf
from gpytorch.kernels import GridInterpolationKernel, MaternKernel, ScaleKernel

from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
from gpytorch.priors import GammaPrior
from gpytorch.settings import (
    max_cholesky_size,
    detach_test_caches,
    use_toeplitz,
    max_root_decomposition_size,
    fast_pred_var,
)
from pandas import DataFrame

from volatilitygp.models import FixedNoiseVariationalGP

from data import prepare_data

# script is based off of https://github.com/wjmaddox/online_gp/blob/main/experiments/active_learning/qnIPV_experiment.py


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_init", type=int, help="(int) number of initial points", default=10,
    )
    parser.add_argument("--num_total", type=int, default=1000000)
    parser.add_argument("--data_loc", type=str, default="./malaria_df.hdf5")
    parser.add_argument("--sketch_size", type=int, default=512)
    parser.add_argument("--cholesky_size", type=int, default=901)
    parser.add_argument("--output", type=str, default="./malaria_output.pt")
    parser.add_argument("--exact", action="store_true")
    parser.add_argument("--toeplitz", action="store_true")
    parser.add_argument("--reset_training_data", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--model", type=str, choices=["exact", "wiski", "svgp"])
    parser.add_argument("--num_steps", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main(args):
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    init_dict, train_dict, test_dict = prepare_data(
        args.data_loc, args.num_init, args.num_total, test_is_year=False, seed=args.seed,
    )
    init_x, init_y, init_y_var = (
        init_dict["x"].to(device),
        init_dict["y"].to(device),
        init_dict["y_var"].to(device),
    )
    train_x, train_y, train_y_var = (
        train_dict["x"].to(device),
        train_dict["y"].to(device),
        train_dict["y_var"].to(device),
    )
    test_x, test_y, test_y_var = (
        test_dict["x"].to(device),
        test_dict["y"].to(device),
        test_dict["y_var"].to(device),
    )

    if args.model == "wiski":

        def model_generator(init_x, init_y, init_y_var):
            return FixedNoiseGP(
                init_x,
                init_y.view(-1, 1),
                init_y_var.view(-1, 1),
                GridInterpolationKernel(
                    base_kernel=ScaleKernel(
                        MaternKernel(
                            ard_num_dims=2,
                            nu=0.5,
                            lengthscale_prior=GammaPrior(3.0, 6.0),
                        ),
                        outputscale_prior=GammaPrior(2.0, 0.15),
                    ),
                    grid_size=30,
                    num_dims=2,
                    grid_bounds=torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
                ),
            ).to(device)

        # mll_type = lambda x, y: BatchedWoodburyMarginalLogLikelihood(x, y, clear_caches_every_iteration=True)
        mll_kwargs = {}
        # mll_type = BatchedWoodburyMarginalLogLikelihood
        mll_type = ExactMarginalLogLikelihood

    elif args.model == "exact":

        def model_generator(init_x, init_y, init_y_var):
            return FixedNoiseGP(
                init_x,
                init_y.view(-1, 1),
                init_y_var.view(-1, 1),
                ScaleKernel(
                    MaternKernel(
                        ard_num_dims=2, nu=0.5, lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
            ).to(device)

        mll_kwargs = {}
        mll_type = ExactMarginalLogLikelihood
    elif args.model == "svgp":
        # likelihood = FixedNoiseGaussianLikelihood(init_y_var)
        def model_generator(init_x, init_y, init_y_var):
            return FixedNoiseVariationalGP(
                init_points=init_x,
                init_targets=init_y.view(-1),
                init_y_var=init_y_var.view(-1),
                num_inducing=900,
                covar_module=ScaleKernel(
                    MaternKernel(
                        ard_num_dims=2, nu=0.5, lengthscale_prior=GammaPrior(3.0, 6.0),
                    ),
                    outputscale_prior=GammaPrior(2.0, 0.15),
                ),
            ).to(device)

        mll_kwargs = {"num_data": init_x.shape[-2]}
        mll_type = VariationalELBO

    model = model_generator(init_x, init_y, init_y_var)
    mll = mll_type(model.likelihood, model, **mll_kwargs)

    if args.model == "wiski":
        # was seeing stability issues with the higher lr
        lr = 0.05
    else:
        lr = 0.1

    print("---- Fitting initial model ----")
    start = time.time()

    model.train()
    model.zero_grad()
    # with max_cholesky_size(args.cholesky_size), skip_logdet_forward(True), \
    #       use_toeplitz(args.toeplitz), max_root_decomposition_size(args.sketch_size):
    fit_gpytorch_torch(mll, options={"lr": lr, "maxiter": 1000})
    end = time.time()
    print("Elapsed fitting time: ", end - start)
    print("Named parameters: ", list(model.named_parameters()))

    print("--- Now computing initial RMSE")
    model.eval()
    with gpytorch.settings.skip_posterior_variances(True):
        test_pred = model(test_x)
        pred_rmse = ((test_pred.mean.view(-1) - test_y.view(-1)) ** 2).mean().sqrt()

    print("---- Initial RMSE: ", pred_rmse.item())

    all_outputs = []
    start_ind = init_x.shape[0]
    end_ind = int(start_ind + args.batch_size)
    for step in range(args.num_steps):
        torch.cuda.empty_cache()
        print("memory used: ", torch.cuda.memory_allocated(device) / (1024 ** 3))
        if step > 0 and step % 5 == 0:
            print("Beginning step ", step)

        total_time_step_start = time.time()

        if step > 0:
            print("---- Fitting model ----")
            start = time.time()

            model.train()
            model.zero_grad()
            if args.model == "svgp":
                mll_kwargs = {"num_data": mll_kwargs["num_data"] + args.batch_size}
            mll = mll_type(model.likelihood, model, **mll_kwargs)
            # with skip_logdet_forward(True), max_root_decomposition_size(
            #         args.sketch_size
            #     ), max_cholesky_size(args.cholesky_size), use_toeplitz(
            #         args.toeplitz
            #     ):
            fit_gpytorch_torch(mll, options={"lr": lr * (0.999 ** step), "maxiter": 300})

            model.zero_grad()
            end = time.time()
            print("Elapsed fitting time: ", end - start)
            # print("Named parameters: ", list(model.named_parameters()))

        print("--- Now computing updated RMSE")
        with torch.no_grad():
            print(model.train_inputs[0].shape, test_x.shape)
            test_pred = model.posterior(test_x)
            # model.eval()
            # test_pred = model(test_x)
            pred_rmse = (
                ((test_pred.mean.view(-1) - test_y.view(-1)) ** 2).mean().sqrt().cpu()
            )
            pred_avg_variance = test_pred.variance.mean().cpu()

        if not args.random:
            #             if args.model == "wiski":
            #                 botorch_model = OnlineSKIBotorchModel(model = model)
            #             else:
            botorch_model = model

            # qmc_sampler = SobolQMCNormalSampler(num_samples=4)

            bounds = torch.stack([torch.zeros(2), torch.ones(2)]).to(device)
            qnipv = qNIPV(
                model=botorch_model,
                mc_points=test_x,
                # sampler=qmc_sampler,
            )

            # with use_toeplitz(args.toeplitz), root_pred_var(True), fast_pred_var(True):
            candidates, acq_value = optimize_acqf(
                acq_function=qnipv,
                bounds=bounds,
                q=args.batch_size,
                num_restarts=4,
                raw_samples=64,  # used for intialization heuristic
                options={"batch_limit": 5, "maxiter": 200},
            )
        else:
            candidates = torch.rand(
                args.batch_size, train_x.shape[-1], device=device, dtype=train_x.dtype
            )
            acq_value = torch.zeros(1)
            model.eval()
            _ = model(test_x[:10])  # to init caches

        print("---- Finished optimizing; now querying dataset ---- ")
        with torch.no_grad():
            covar_dists = model.covar_module(candidates, train_x)
            nearest_points = covar_dists.evaluate().argmax(dim=-1)
            new_x = train_x[nearest_points]
            new_y = train_y[nearest_points]
            new_y_var = train_y_var[nearest_points]

            todrop = torch.tensor([x in nearest_points for x in range(train_x.shape[0])])
            train_x, train_y, train_y_var = (
                train_x[~todrop],
                train_y[~todrop],
                train_y_var[~todrop],
            )
            print("New train_x shape", train_x.shape)
            print("--- Now updating model with simulator ----")
            updated_x = torch.cat((new_x, model.train_inputs[0]))
            updated_y = torch.cat((new_y, model.train_targets))
            updated_y_var = torch.cat((new_y_var, model.likelihood.noise))
            model.train()
            model._memoize_cache = {}
            del model

            model = model_generator(updated_x, updated_y, updated_y_var,)

        total_time_step_elapsed_time = time.time() - total_time_step_start
        step_output_list = [
            total_time_step_elapsed_time,
            acq_value.item(),
            pred_rmse.item(),
            pred_avg_variance.item(),
        ]
        print("Step RMSE: ", pred_rmse)
        all_outputs.append(step_output_list)

        start_ind = end_ind
        end_ind = int(end_ind + args.batch_size)

    output_dict = {
        "model_state_dict": model.cpu().state_dict(),
        "queried_points": {
            "x": model.cpu().train_inputs[0],
            "y": model.cpu().train_targets,
        },
        "results": DataFrame(all_outputs),
    }
    torch.save(output_dict, args.output)


if __name__ == "__main__":
    args = parse()
    with fast_pred_var(True), use_toeplitz(args.toeplitz), detach_test_caches(
        True
    ), max_cholesky_size(args.cholesky_size), max_root_decomposition_size(
        args.sketch_size
    ):  # , \
        # root_pred_var(True):
        main(args)
