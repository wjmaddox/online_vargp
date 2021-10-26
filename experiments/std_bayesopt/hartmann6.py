import time
import torch

from botorch.acquisition.objective import ConstrainedMCObjective
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling.samplers import SobolQMCNormalSampler

from utils import (
    generate_initial_data,
    initialize_model,
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
)

# script taken from https://botorch.org/tutorials/closed_loop_botorch_only


def main(
    seed: int = 0,
    method: str = "variational",
    batch_size: int = 3,
    n_batch: int = 50,
    mc_samples: int = 256,
    num_init: int = 10,
    noise_se: float = 0.5,
    dtype: str = "double",
    verbose: bool = True,
    output: str = None,
    problem: str = None,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(seed)

    from botorch.test_functions import Hartmann

    neg_hartmann6 = Hartmann(negate=True)
    bounds = torch.tensor([[0.0] * 6, [1.0] * 6], device=device, dtype=dtype)

    def outcome_constraint(X):
        """L1 constraint; feasible if less than or equal to zero."""
        return X.sum(dim=-1) - 3

    def weighted_obj(X):
        """Feasibility weighted objective; zero if not feasible."""
        return neg_hartmann6(X) * (outcome_constraint(X) <= 0).type_as(X)

    def obj_callable(Z):
        return Z[..., 0]

    def constraint_callable(Z):
        return Z[..., 1]

    # define a feasibility-weighted objective for optimization
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable, constraints=[constraint_callable],
    )

    best_observed_ei, best_observed_nei, best_observed_gibbon, best_random = (
        [],
        [],
        [],
        [],
    )

    train_yvar = torch.tensor(noise_se ** 2, device=device, dtype=dtype)

    # call helper functions to generate initial training data and initialize model
    (
        train_x_ei,
        train_obj_ei,
        train_con_ei,
        best_observed_value_ei,
    ) = generate_initial_data(
        num_init, neg_hartmann6, outcome_constraint, weighted_obj, noise_se, device, dtype
    )
    mll_ei, model_ei = initialize_model(
        train_x_ei, train_obj_ei, train_con_ei, train_yvar, method=method
    )

    train_x_nei, train_obj_nei, train_con_nei = train_x_ei, train_obj_ei, train_con_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(
        train_x_nei, train_obj_nei, train_con_nei, train_yvar, method=method
    )

    train_x_gibbon, train_obj_gibbon, train_con_gibbon = (
        train_x_ei,
        train_obj_ei,
        train_con_ei,
    )
    best_observed_value_gibbon = best_observed_value_ei
    mll_gibbon, model_gibbon = initialize_model(
        train_x_gibbon, train_obj_gibbon, train_con_gibbon, train_yvar, method=method
    )

    best_observed_ei.append(best_observed_value_ei)
    best_observed_nei.append(best_observed_value_nei)
    best_observed_gibbon.append(best_observed_value_gibbon)
    best_random.append(best_observed_value_ei)

    # run N_BATCH rounds of BayesOpt after the initial random batch
    for iteration in range(1, n_batch + 1):

        t0 = time.time()

        # fit the models
        optimizer_kwargs = {"maxiter": 1000}
        fit_gpytorch_torch(mll_ei, options=optimizer_kwargs)
        fit_gpytorch_torch(mll_nei, options=optimizer_kwargs)
        fit_gpytorch_torch(mll_gibbon, options=optimizer_kwargs)

        # define the qEI and qNEI acquisition modules using a QMC sampler
        qmc_sampler = SobolQMCNormalSampler(num_samples=mc_samples)

        # for best_f, we use the best observed noisy values as an approximation
        qEI = qExpectedImprovement(
            model=model_ei,
            best_f=(train_obj_ei * (train_con_ei <= 0).to(train_obj_ei)).max(),
            sampler=qmc_sampler,
            objective=constrained_obj,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
            objective=constrained_obj,
        )

        candidate_set = torch.rand(1000, train_x_ei.shape[-1], device=device, dtype=dtype)
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        qGIBBON = qKnowledgeGradient(
            model_gibbon,
            objective=constrained_obj,
            current_value=best_observed_value_gibbon,
        )

        # optimize and get new observation
        optimize_acqf_kwargs = {
            "bounds": bounds,
            "BATCH_SIZE": batch_size,
            "fn": neg_hartmann6,
            "outcome_constraint": outcome_constraint,
            "noise_se": noise_se,
        }
        new_x_ei, new_obj_ei, new_con_ei = optimize_acqf_and_get_observation(
            qEI, **optimize_acqf_kwargs
        )
        new_x_nei, new_obj_nei, new_con_nei = optimize_acqf_and_get_observation(
            qNEI, **optimize_acqf_kwargs
        )
        new_x_gibbon, new_obj_gibbon, new_con_gibbon = optimize_acqf_and_get_observation(
            qGIBBON, **optimize_acqf_kwargs
        )

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei])
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei])
        train_con_ei = torch.cat([train_con_ei, new_con_ei])

        train_x_nei = torch.cat([train_x_nei, new_x_nei])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei])
        train_con_nei = torch.cat([train_con_nei, new_con_nei])

        train_x_gibbon = torch.cat([train_x_gibbon, new_x_gibbon])
        train_obj_gibbon = torch.cat([train_obj_gibbon, new_obj_gibbon])
        train_con_gibbon = torch.cat([train_con_gibbon, new_con_gibbon])

        # update progress
        best_random = update_random_observations(batch_size, best_random, weighted_obj)
        best_value_ei = weighted_obj(train_x_ei).max().item()
        best_value_nei = weighted_obj(train_x_nei).max().item()
        best_value_gibbon = weighted_obj(train_x_gibbon).max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)
        best_observed_gibbon.append(best_value_gibbon)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei,
            train_obj_ei,
            train_con_ei,
            train_yvar,
            method=method,
            # model_ei.state_dict(),
        )
        mll_nei, model_nei = initialize_model(
            train_x_nei,
            train_obj_nei,
            train_con_nei,
            train_yvar,
            # model_nei.state_dict(),
            method=method,
        )
        mll_gibbon, model_gibbon = initialize_model(
            train_x_gibbon,
            train_obj_gibbon,
            train_con_gibbon,
            train_yvar,
            # model_nei.state_dict(),
            method=method,
        )
        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (random, qEI, qNEI, qGIBBON) = "
                f"({max(best_random):>4.2f}, {best_value_ei:>4.2f}, {best_value_nei:>4.2f}, {best_value_gibbon:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")

    output_dict = {
        "rnd": best_random,
        "ei": best_observed_ei,
        "nei": best_observed_nei,
        "gibbon": best_observed_gibbon,
    }
    return output_dict


if __name__ == "__main__":
    args = parse()
    output_dict = main(**vars(args))
    torch.save({"pars": vars(args), "results": output_dict}, args.output)
