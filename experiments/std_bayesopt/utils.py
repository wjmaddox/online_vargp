import argparse

import torch

from botorch.exceptions import BadInitialCandidatesWarning

import warnings


warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

from botorch.optim import optimize_acqf

from volatilitygp.models import SingleTaskVariationalGP, ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood, PredictiveLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.priors import GammaPrior
from volatilitygp.mlls import PatchedVariationalELBO as VariationalELBO
from gpytorch.mlls import PredictiveLogLikelihood
from volatilitygp.likelihoods import PoissonLikelihood
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from torch.distributions import Poisson

from gpytorch.kernels import ScaleKernel, MaternKernel

from botorch.models import SingleTaskGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    parser.add_argument("--problem", type=str, default="hartmann6")
    parser.add_argument("--n_batch", type=int, default=50)
    parser.add_argument("--num_init", type=int, default=10)
    parser.add_argument("--mc_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--method", type=str, default="variational")
    return parser.parse_args()


def get_var_model(
    x,
    y,
    yvar,
    num_inducing=25,
    is_poisson=True,
    use_input_transform=False,
    use_outcome_transform=False,
    **kwargs
):

    if is_poisson:
        likelihood = PoissonLikelihood()
    elif yvar is None:
        likelihood = GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2))
    else:
        likelihood = None
    # likelihood = PoissonLikelihood() if is_poisson else None

    model = SingleTaskVariationalGP(
        init_points=x,
        num_inducing=num_inducing,
        likelihood=likelihood,
        covar_module=ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5, ard_num_dims=x.shape[-1], lengthscale_prior=GammaPrior(3.0, 6.0),
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        ),
        init_targets=y,
        outcome_transform=Standardize(y.shape[-1]) if use_outcome_transform else None,
        # outcome_transform=Standardize(y.shape[-1]) if not is_poisson else None,
        input_transform=Normalize(x.shape[-1]) if use_input_transform else None,
        train_inputs=x,
        train_targets=y,
        **kwargs,
    ).to(x)
    # model.train_inputs = [x]
    # model.train_targets = y.view(-1)

    if not is_poisson and yvar is not None:
        model.likelihood.raw_noise.detach_()
        model.likelihood.noise = yvar.item()

    return model


def get_exact_model(
    x, y, yvar, use_input_transform=False, use_outcome_transform=False, **kwargs
):
    model = SingleTaskGP(
        x,
        y,
        likelihood=GaussianLikelihood(noise_constraint=Interval(5e-4, 0.2))
        if yvar is None
        else None,
        outcome_transform=Standardize(y.shape[-1]) if use_outcome_transform else None,
        input_transform=Normalize(x.shape[-1]) if use_input_transform else None,
    ).to(x)
    if yvar is not None:
        model.likelihood.raw_noise.detach_()
        model.likelihood.noise = yvar.item()
    return model


def generate_initial_data(
    n, fn, outcome_constraint, weighted_obj, NOISE_SE, device, dtype, is_poisson=False
):
    # generate training data
    train_x = torch.rand(n, fn.dim, device=device, dtype=dtype)
    exact_obj = fn(train_x).unsqueeze(-1)  # add output dimension
    exact_con = outcome_constraint(train_x).unsqueeze(-1)  # add output dimension
    train_obj = exact_obj + NOISE_SE * torch.randn_like(exact_obj)
    if is_poisson:
        train_obj = Poisson(train_obj.exp()).sample()

    train_con = exact_con + NOISE_SE * torch.randn_like(exact_con)
    best_observed_value = weighted_obj(train_x).max().item()
    return train_x, train_obj, train_con, best_observed_value


def initialize_model(
    train_x, train_obj, train_con, train_yvar, state_dict=None, method="variational"
):
    # define models for objective and constraint
    if method == "variational":
        model_obj = get_var_model(train_x, train_obj, train_yvar, is_poisson=False)
        model_con = get_var_model(train_x, train_con, train_yvar, is_poisson=False)
        kwargs = {
            "mll_cls": lambda l, m: VariationalELBO(l, m, num_data=train_x.shape[-2])
        }
    elif method == "exact":
        model_obj = get_exact_model(train_x, train_obj, train_yvar)
        model_con = get_exact_model(train_x, train_con, train_yvar)
        kwargs = {}

    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model, **kwargs)
    # mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def initialize_model_poisson(
    train_x, train_obj, train_con, train_yvar, state_dict=None, method="variational"
):
    # define models for objective and constraint
    if method == "variational":
        model_obj = get_var_model(train_x, train_obj, train_yvar, is_poisson=True)
        model_con = get_var_model(train_x, train_con, train_yvar, is_poisson=False)
        kwargs = {
            "mll_cls": lambda l, m: VariationalELBO(l, m, num_data=train_x.shape[-2])
        }
    elif method == "exact":
        model_obj = get_exact_model(train_x, train_obj, train_yvar)
        model_con = get_exact_model(train_x, train_con, train_yvar)
        kwargs = {}

    # combine into a multi-output GP model
    model = ModelListGP(model_obj, model_con)
    mll = SumMarginalLogLikelihood(model.likelihood, model, **kwargs)
    # mll = SumMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(
    acq_func,
    bounds,
    BATCH_SIZE,
    fn,
    outcome_constraint=None,
    noise_se=0.0,
    NUM_RESTARTS=10,
    RAW_SAMPLES=512,
    is_poisson=False,
    sequential=False,
):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=BATCH_SIZE,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200},
        sequential=sequential,
    )
    # observe new values
    new_x = candidates.detach()
    exact_obj = fn(new_x).unsqueeze(-1)  # add output dimension
    if outcome_constraint is not None:
        exact_con = outcome_constraint(new_x).unsqueeze(-1)  # add output dimension
    new_obj = exact_obj + noise_se * torch.randn_like(exact_obj)
    if is_poisson:
        new_obj = Poisson(new_obj.exp()).sample()
    if outcome_constraint is not None:
        new_con = exact_con + noise_se * torch.randn_like(exact_con)
    else:
        new_con = None
    return new_x, new_obj, new_con


def initialize_model_unconstrained(
    train_x,
    train_obj,
    train_yvar=None,
    state_dict=None,
    method="variational",
    loss="elbo",
    **kwargs
):
    # define models for objective and constraint
    if method == "variational":
        model_obj = get_var_model(
            train_x, train_obj, train_yvar, is_poisson=False, **kwargs
        )
        if loss == "elbo":
            mll = VariationalELBO(
                model_obj.likelihood, model_obj, num_data=train_x.shape[-2]
            )
        elif loss == "pll":
            mll = PredictiveLogLikelihood(
                model_obj.likelihood, model_obj, num_data=train_x.shape[-2]
            )
    elif method == "exact":
        model_obj = get_exact_model(train_x, train_obj, train_yvar, **kwargs)
        mll = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)

    model_obj = model_obj.to(train_x.device)
    # load state dict if it is passed
    if state_dict is not None:
        model_obj.load_state_dict(state_dict)
    return mll, model_obj


def update_random_observations(BATCH_SIZE, best_random, problem=lambda x: x, dim=6):
    """Simulates a random policy by taking a the current list of best values observed randomly,
    drawing a new random point, observing its value, and updating the list.
    """
    rand_x = torch.rand(BATCH_SIZE, dim)
    next_random_best = problem(rand_x).max().item()
    best_random.append(max(best_random[-1], next_random_best))
    return best_random
