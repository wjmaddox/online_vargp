import argparse
import time
import numpy as np
import torch
import sys

from botorch.optim.fit import fit_gpytorch_torch

sys.path.append("../std_bayesopt/")
from utils import initialize_model_unconstrained as initialize_model

sys.path.append("./")
from trbo import TurboState, update_state, generate_batch


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--problem", type=str, default="rover")
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--num_init", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--method", type=str, default="exact")
    parser.add_argument("--acqf", type=str, default="ts")
    parser.add_argument("--num_inducing", type=int, default=500)
    parser.add_argument("--tree_depth", type=int, default=4)
    parser.add_argument("--use_full", action="store_true")
    parser.add_argument("--dim", type=int, default=30)
    return parser.parse_args()


def main(
    seed: int = 0,
    method: str = "exact",
    batch_size: int = 100,
    n_batch: int = 200,
    num_init: int = 200,
    dtype: str = "double",
    output: str = None,
    problem: str = None,
    acqf: str = "ts",
    use_full: bool = False,
    num_inducing: int = 500,
    loss: str = "pll",
    tree_depth: int = 4,
    dim: int = 30,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(seed)

    NUM_RESTARTS = 5
    RAW_SAMPLES = 256

    if problem == "rover":
        from rover_function import create_large_domain

        def l2cost(x, point):
            return 10 * np.linalg.norm(x - point, 1)

        domain = create_large_domain(
            force_start=False,
            force_goal=False,
            start_miss_cost=l2cost,
            goal_miss_cost=l2cost,
            n_points=dim,
        )
        n_points = domain.traj.npoints

        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        # 5 is the max achievable, this is the offset
        bounded_fn_callable = lambda X: torch.stack(
            [torch.tensor(domain(x.cpu().numpy()) + 5.0) for x in X]
        ).to(X)
        # map into bounds, thanks..
        # we need to map this from [0, 1]^d -> [-0.1, 1.1]^d
        fn_callable = lambda X: bounded_fn_callable(X * 1.2 - 0.1)
        bounds = torch.tensor(raw_x_range, dtype=dtype, device=device)
        bounds = torch.zeros(2, raw_x_range.shape[-1], dtype=dtype, device=device)
        bounds[1] = 1.0
        dim = bounds.shape[-1]

    num_batches = 0
    N_CANDIDATES = min(5000, max(2000, 200 * dim))
    timing_list = []

    # test_x = torch.rand(5 * num_init, bounds.shape[-1], device=device, dtype=dtype) * \
    #     (bounds[1] - bounds[0]) + bounds[0]
    # test_obj = fn_callable(test_x)

    while num_batches < n_batch + 2:

        next_x = (
            torch.rand(num_init, bounds.shape[-1], device=device, dtype=dtype)
            * (bounds[1] - bounds[0])
            + bounds[0]
        )
        next_obj = fn_callable(next_x).unsqueeze(-1)

        if use_full and num_batches > 0:
            train_x = torch.cat((train_x, next_x))
            train_obj = torch.cat((train_obj, next_obj))
        else:
            # normal trbo
            train_x = next_x
            train_obj = next_obj

        num_batches += 1
        if num_batches == 1:
            best_observed_value = [train_obj.max().item()]

        state = TurboState(
            bounds.shape[-1], batch_size=batch_size,  # best_value=train_obj.max().item()
        )

        # we loop turbo until the state restarts
        while not state.restart_triggered:
            start = time.time()

            # generate a new model
            mll_gibbon, model = initialize_model(
                train_x,
                train_obj,
                None,
                method=method,
                use_input_transform=False,
                use_outcome_transform=True,
                num_inducing=num_inducing,
                loss=loss,
            )
            # fit the models
            optimizer_kwargs = {"maxiter": 1000}
            fit_gpytorch_torch(mll_gibbon, options=optimizer_kwargs)

            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=train_x,
                Y=train_obj,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf=acqf,
                tree_depth=tree_depth,
            )
            Y_next = fn_callable(X_next).unsqueeze(-1)

            # Update state
            state = update_state(state=state, Y_next=Y_next)

            end = time.time()
            # Append data
            train_x = torch.cat((train_x, X_next), dim=0)
            train_obj = torch.cat((train_obj, Y_next), dim=0)

            mem = torch.cuda.memory_allocated(device) / 1024 ** 3

            # Print current status
            print(
                f"Iter {num_batches}: {len(train_x)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e},"
                f" Mem used {mem:.2e}"
            )

            best_observed_value.append(state.best_value)
            num_batches += 1

            #             with torch.no_grad():
            #                 mse = ((model.posterior(test_x).mean.view(-1) - test_obj.view(-1))**2).mean()
            #                 print("test set mse: ", mse.item())

            timing_list.append(end - start)
            if num_batches % 10 == 0:
                output_dict = {"trbo": best_observed_value}
                torch.save(
                    {"iters": num_batches, "times": timing_list, "results": output_dict},
                    output,
                )

            if num_batches > n_batch + 2:
                break

    output_dict = {"trbo": best_observed_value}
    return output_dict, timing_list


if __name__ == "__main__":
    args = parse()
    output_dict, timing_list = main(**vars(args))
    torch.save(
        {"pars": vars(args), "times": timing_list, "results": output_dict}, args.output
    )
