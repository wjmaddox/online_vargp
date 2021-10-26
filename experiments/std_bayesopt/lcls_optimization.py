import time
import torch

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)

from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling.samplers import SobolQMCNormalSampler

from utils import (
    parse,
    optimize_acqf_and_get_observation,
    update_random_observations,
)

from utils import initialize_model_unconstrained as initialize_model


def prepare_lcls_function(num_init=100, noise=0.01, rel_path="./weighted_gp_benchmark/"):
    # copied over from the setup from mcintire
    # UAI '16
    num_train = num_init
    import sys
    import numpy as np
    import pandas as pd

    sys.path.append("../../../bayes-opt")
    # this link is https://github.com/ermongroup/bayes-opt/
    from BasicInterfaces import GPint
    from GPtools import vify
    from OnlineGP import OGP
    from SPGPmodel import SPGP

    np.random.seed(1)

    # load data
    data = pd.read_csv(rel_path + "/data.csv")

    # filter 'bad' y-values
    dt = data[(data.iloc[:, 1] > 0.2) & (data.iloc[:, 1] < 6.0)]

    # get list of controlled variables
    ctrl = [x for x in data.columns if x[-5:] == "BCTRL"]

    X = dt[ctrl]
    y = dt.iloc[:, 1]

    # set up data from a given event
    event_energy = 11.45
    Xsm = X.loc[X.iloc[:, 0] == event_energy, :]
    Xsm = Xsm.iloc[50:2050, 1:]
    Ysm = np.array(dt.loc[X.iloc[:, 0] == event_energy, dt.columns[1]])
    Ysm = Ysm[50:2050]
    XYsm = Xsm.copy()
    XYsm["y"] = Ysm
    mins = Xsm.min(axis=0)
    maxs = Xsm.max(axis=0)

    # bound the acquisition: leads to better performance and lessens
    #   the improvements from weighting
    # bnds = tuple([(mins[i],maxs[i]) for i in range(len(mins))])
    # build a sparse GP and optimize its hyperparameters to use for online GP
    #   -  in practice we want to choose hyperparameters more intelligently
    print(np.var(np.array(XYsm), 0), np.array(XYsm).shape)

    hprior = SPGP()
    hprior.fit(Xsm, Ysm, 300)
    data_hyps = hprior.hyps

    prior = OGP(16, data_hyps, weighted=False, maxBV=600, prmean=1)
    prior.fit(Xsm, Ysm)

    intfc2 = GPint(vify(Xsm, 0), prior)

    # need initial training or complex prior mean function to guide optimization
    train = XYsm.iloc[-1000:].sample(n=num_train)
    train.iloc[:, -1] += noise * np.random.randn(num_train)

    class LCLSProblem:
        def __call__(self, X):
            yvals = []
            for x in X:
                self.interface.setX(x.cpu().numpy())
                _, y = self.interface.getState()
                yvals.append(torch.tensor(y).view(-1))
            return torch.stack(yvals).reshape(*X.shape[:-1])

        interface = intfc2
        bounds = torch.tensor(
            tuple([(1.0 * mins[i], 1.0 * maxs[i]) for i in range(len(mins))])
        )

    train_x = torch.tensor(train.iloc[:, :-1].values)
    train_y = torch.tensor(train.iloc[:, -1].values).view(-1, 1)

    return LCLSProblem(), train_x, train_y


def main(
    seed: int = 0,
    method: str = "variational",
    batch_size: int = 3,
    n_batch: int = 50,
    mc_samples: int = 256,
    num_init: int = 100,
    noise_se: float = 0.1,
    dtype: str = "double",
    verbose: bool = True,
    output: str = None,
    problem: str = None,
):
    dtype = torch.double if dtype == "double" else torch.float
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.random.manual_seed(seed)
    lcls_fn, init_x, init_y = prepare_lcls_function(num_init=num_init)
    bounds = lcls_fn.bounds.to(device=device, dtype=dtype).t()
    print(bounds, bounds.shape)

    best_observed_ei, best_observed_nei, best_observed_gibbon, best_random = (
        [],
        [],
        [],
        [],
    )

    train_yvar = torch.tensor(noise_se ** 2, device=device, dtype=dtype)
    train_x_ei = init_x.to(train_yvar)
    train_obj_ei = init_y.to(train_yvar)
    # best_observed_value_ei = train_obj_ei.max()
    best_observed_value_ei = 0.0

    # call helper functions to generate initial training data and initialize model
    mll_ei, model_ei = initialize_model(
        train_x_ei, train_obj_ei, train_yvar, method=method
    )

    train_x_nei, train_obj_nei = train_x_ei, train_obj_ei  # , train_con_ei
    best_observed_value_nei = best_observed_value_ei
    mll_nei, model_nei = initialize_model(
        train_x_nei, train_obj_nei, train_yvar, method=method
    )

    train_x_gibbon, train_obj_gibbon = train_x_ei, train_obj_ei  # , con_ei
    best_observed_value_gibbon = best_observed_value_ei
    mll_gibbon, model_gibbon = initialize_model(
        train_x_gibbon, train_obj_gibbon, train_yvar, method=method
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
            best_f=train_obj_ei.max().item(),
            sampler=qmc_sampler,
            # objective=constrained_obj,
        )

        qNEI = qNoisyExpectedImprovement(
            model=model_nei,
            X_baseline=train_x_nei,
            sampler=qmc_sampler,
            # objective=constrained_obj,
        )

        candidate_set = torch.rand(1000, train_x_ei.shape[-1], device=device, dtype=dtype)
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        qGIBBON = qKnowledgeGradient(
            model_gibbon,
            # objective=constrained_obj,
            current_value=train_obj_gibbon.max().item(),
        )

        # optimize and get new observation
        optimize_acqf_kwargs = {
            "bounds": bounds,
            "BATCH_SIZE": batch_size,
            "fn": lcls_fn,
            # "outcome_constraint": outcome_constraint,
            "noise_se": noise_se,
        }
        new_x_ei, new_obj_ei, _ = optimize_acqf_and_get_observation(
            qEI, **optimize_acqf_kwargs
        )
        new_x_nei, new_obj_nei, _ = optimize_acqf_and_get_observation(
            qNEI, **optimize_acqf_kwargs
        )
        new_x_gibbon, new_obj_gibbon, _ = optimize_acqf_and_get_observation(
            qGIBBON, **optimize_acqf_kwargs
        )

        # update training points
        train_x_ei = torch.cat([train_x_ei, new_x_ei.to(train_x_ei)])
        print(train_obj_ei.shape, new_obj_ei)
        train_obj_ei = torch.cat([train_obj_ei, new_obj_ei.to(train_x_ei)])

        train_x_nei = torch.cat([train_x_nei, new_x_nei.to(train_x_ei)])
        train_obj_nei = torch.cat([train_obj_nei, new_obj_nei.to(train_x_ei)])

        train_x_gibbon = torch.cat([train_x_gibbon, new_x_gibbon.to(train_x_ei)])
        train_obj_gibbon = torch.cat([train_obj_gibbon, new_obj_gibbon.to(train_x_ei)])

        # update progress
        # we only consider the new values
        best_random = update_random_observations(
            batch_size, best_random, lcls_fn, train_x_ei.shape[-1]
        )
        best_value_ei = (train_obj_ei[num_init:]).max().item()
        best_value_nei = (train_obj_nei[num_init:]).max().item()
        best_value_gibbon = (train_obj_gibbon[num_init:]).max().item()
        best_observed_ei.append(best_value_ei)
        best_observed_nei.append(best_value_nei)
        best_observed_gibbon.append(best_value_gibbon)

        # reinitialize the models so they are ready for fitting on next iteration
        # use the current state dict to speed up fitting
        mll_ei, model_ei = initialize_model(
            train_x_ei, train_obj_ei, train_yvar, method=method,
        )
        mll_nei, model_nei = initialize_model(
            train_x_nei, train_obj_nei, train_yvar, method=method,
        )
        mll_gibbon, model_gibbon = initialize_model(
            train_x_gibbon, train_obj_gibbon, train_yvar, method=method,
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
    data_dict = {
        "ei": train_x_ei.cpu(),
        "nei": train_x_nei.cpu(),
        "gibbon": train_x_gibbon.cpu(),
    }
    return output_dict, data_dict


if __name__ == "__main__":
    args = parse()
    output_dict, data_dict = main(**vars(args))
    torch.save(
        {"pars": vars(args), "results": output_dict, "data": data_dict}, args.output
    )
