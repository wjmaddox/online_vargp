import torch

import gpytorch

from botorch.acquisition import qNoisyExpectedImprovement, qKnowledgeGradient
from botorch.optim import optimize_acqf

import sys

sys.path.append("./")
from utils import (
    generate_data,
    generate_comparisons,
    make_new_data,
    utility,
    init_and_fit_model,
    parse,
)


def main(
    seed: int = 0,
    dim: int = 10,
    num_restarts: int = 3,
    raw_samples: int = 128,
    batch_size: int = 3,
    batch_comparisons: int = 3,
    n_batch: int = 50,
    output: str = "results.pt",
    method: str = "laplace",
    noise: float = 0.1,
):
    dtype = torch.double

    algos = ["qNEI", "rand"]
    if method == "variational":
        algos.append(("qKG"))

    # initial evals
    best_vals = {}  # best observed values
    for algo in algos:
        best_vals[algo] = []

    data = {}
    models = {}

    torch.random.manual_seed(seed)
    # Create initial data
    init_X, init_y = generate_data(3 * batch_size, dim=dim)
    comparisons = generate_comparisons(init_y, 6 * batch_comparisons, noise=noise)

    # test_X, test_y = generate_data(1000, dim=dim)
    # X are within the unit cube
    bounds = torch.stack([torch.zeros(dim), torch.ones(dim)]).to(dtype)

    for algo in algos:
        best_vals[algo].append([])
        data[algo] = (init_X, comparisons)
        _, models[algo] = init_and_fit_model(init_X, comparisons, method)
        model = models[algo]

        best_next_y = utility(init_X).max().item()
        best_vals[algo][-1].append(best_next_y)

    # we make additional n_batch comparison queries after the initial observation
    for j in range(1, n_batch + 1):
        for algo in algos:
            if algo != "rand":
                if algo == "qNEI":
                    # create the acquisition function object
                    acq_func = qNoisyExpectedImprovement(
                        model=model, X_baseline=data["qNEI"][0].float()
                    )
                elif algo == "qKG":
                    acq_func = qKnowledgeGradient(
                        model, current_value=utility(data["qKG"][0]).max().item()
                    )

                with gpytorch.settings.cholesky_jitter(1e-4):
                    # optimize and get new observation
                    next_X, acq_val = optimize_acqf(
                        acq_function=acq_func,
                        bounds=bounds,
                        q=batch_size,
                        num_restarts=num_restarts,
                        raw_samples=raw_samples,
                    )
            else:
                # randomly sample data
                next_X, _ = generate_data(batch_size, dim=dim)

            # update data
            X, comps = data[algo]
            X, comps = make_new_data(X, next_X, comps, batch_comparisons, noise=noise)
            data[algo] = (X, comps)

            # refit models
            model = models[algo]
            _, model = init_and_fit_model(X, comps, method)
            models[algo] = model

            # record the best observed values so far
            max_val = utility(X).max().item()
            best_vals[algo][-1].append(max_val)

        best_kg = max(best_vals["qKG"][-1]) if method == "variational" else 0.0
        best_random = max(best_vals["rand"][-1])
        best_nei = max(best_vals["qNEI"][-1])
        print(
            f"\nBatch {j:>4.2f}: best value (random, qnei, qkg)"
            f"({best_random:>4.2f}, {best_nei:>4.2f}, {best_kg:>4.2f})",
            end="",
        )

    return {key: best_vals[val] for key, best_vals[val] in zip(algos, best_vals)}


if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
