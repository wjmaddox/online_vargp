import argparse
import time
import torch
import numpy as np

from gpytorch import settings

from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.optim.fit import fit_gpytorch_torch
from botorch.generation import MaxPosteriorSampling

import sys

sys.path.append("../std_bayesopt/")
from utils import (
    optimize_acqf_and_get_observation,
    update_random_observations,
)
from utils import initialize_model_unconstrained as initialize_model

sys.path.append("./")
from trbo import generate_candidates


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--problem", type=str, default="hartmann6")
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--mc_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument(
        "--ts_split", action="store_true", help="devote half batch to ts for qgibbon"
    )
    parser.add_argument("--method", type=str, default="variational")
    return parser.parse_args()


def main(
    seed: int = 0,
    method: str = "variational",
    batch_size: int = 3,
    n_batch: int = 50,
    mc_samples: int = 256,
    num_init: int = 200,
    noise_se: float = 0.5,
    dtype: str = "double",
    verbose: bool = True,
    output: str = None,
    ts_split: bool = False,
    problem: str = None,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(seed)

    if problem == "rover":
        from rover_function import create_large_domain

        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        domain = create_large_domain(
            force_start=False,
            force_goal=False,
            start_miss_cost=l2cost,
            goal_miss_cost=l2cost,
        )
        n_points = domain.traj.npoints

        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        # 5 is the max achievable, this is the offset
        fn_callable = lambda X: torch.stack(
            [torch.tensor(domain(x.cpu().numpy()) + 5.0) for x in X]
        ).to(X)
        bounds = torch.tensor(raw_x_range, dtype=dtype, device=device)

    train_x_gibbon = (
        torch.rand(num_init, bounds.shape[-1], device=device, dtype=dtype)
        * (bounds[1] - bounds[0])
        + bounds[0]
    )
    train_obj_gibbon = fn_callable(train_x_gibbon).unsqueeze(-1)
    best_observed_value_gibbon = train_obj_gibbon.max()

    mll_gibbon, model_gibbon = initialize_model(
        train_x_gibbon,
        train_obj_gibbon,
        None,
        method=method,
        use_input_transform=False,
        use_outcome_transform=True,
        num_inducing=250,
    )

    best_observed_gibbon, best_random = [], []
    best_observed_gibbon.append(best_observed_value_gibbon)
    best_random.append(best_observed_value_gibbon)

    timing_list = []

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):
        print(
            iteration, "memory usage: ", torch.cuda.memory_allocated(device) / (1024 ** 3)
        )
        t0 = time.time()

        # fit the models
        optimizer_kwargs = {"maxiter": 1000}
        fit_gpytorch_torch(mll_gibbon, options=optimizer_kwargs)

        dim = train_x_gibbon.shape[-1]
        n_candidates = min(5000, max(2000, 20 * dim))

        # if using ts_split, we devote half the batch to TS to force qGIBBON to fantasize
        with torch.no_grad():
            if ts_split:
                x_center = train_x_gibbon[train_obj_gibbon.argmax(), :]
                x_cand = generate_candidates(
                    x_center, dim, n_candidates, bounds, dtype, device
                )
                # Sample on the candidate points
                thompson_sampling = MaxPosteriorSampling(
                    model=model_gibbon, replacement=False
                )
                ts_batch = thompson_sampling(x_cand, num_samples=int(batch_size / 2))
                ts_batch_y = fn_callable(ts_batch).unsqueeze(-1)
            else:
                ts_batch = None

        candidate_set = torch.rand(
            n_candidates, train_x_gibbon.shape[-1], device=device, dtype=dtype
        )
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        qGIBBON = qLowerBoundMaxValueEntropy(
            model_gibbon, candidate_set=candidate_set, X_pending=ts_batch
        )

        bs_for_optim = int(batch_size / 2) if ts_split else batch_size

        # optimize and get new observation
        optimize_acqf_kwargs = {
            "bounds": bounds,
            "BATCH_SIZE": bs_for_optim,
            "fn": fn_callable,
            "sequential": True,
        }
        mcs = 100000 if method == "exact" else 800  # default
        fpv = True
        with settings.max_cholesky_size(mcs), settings.fast_pred_var(fpv):
            (
                new_x_gibbon,
                new_obj_gibbon,
                new_con_gibbon,
            ) = optimize_acqf_and_get_observation(qGIBBON, **optimize_acqf_kwargs)

        if ts_split:
            new_x_gibbon = torch.cat([new_x_gibbon, ts_batch])
            new_obj_gibbon = torch.cat([new_obj_gibbon, ts_batch_y])

        train_x_gibbon = torch.cat([train_x_gibbon, new_x_gibbon])
        train_obj_gibbon = torch.cat([train_obj_gibbon, new_obj_gibbon])

        # update progress
        best_random = update_random_observations(
            batch_size, best_random, fn_callable, dim=bounds.shape[1]
        )
        best_value_gibbon = train_obj_gibbon.max().item()
        best_observed_gibbon.append(best_value_gibbon)

        mll_gibbon, model_gibbon = initialize_model(
            train_x_gibbon,
            train_obj_gibbon,
            None,
            method=method,
            use_input_transform=False,
            use_outcome_transform=True,
            num_inducing=250,
        )
        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qGIBBON) = "
                f"({max(best_random):>4.2f}, {best_value_gibbon:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

        timing_list.append(t1 - t0)

        if method != "variational" and iteration % 5 == 0:
            output_dict = {"rnd": best_random, "gibbon": best_observed_gibbon}
            torch.save(
                {"results": output_dict, "times": timing_list, "iteration": iteration},
                output,
            )

    output_dict = {"rnd": best_random, "gibbon": best_observed_gibbon}
    return output_dict, timing_list


if __name__ == "__main__":
    args = parse()
    output_dict, timing_list = main(**vars(args))
    torch.save(
        {"pars": vars(args), "times": timing_list, "results": output_dict}, args.output
    )
