###############################################################################
# Copyright (c) 2019 Uber Technologies, Inc.                                  #
#                                                                             #
# Licensed under the Uber Non-Commercial License (the "License");             #
# you may not use this file except in compliance with the License.            #
# You may obtain a copy of the License at the root directory of this project. #
#                                                                             #
# See the License for the specific language governing permissions and         #
# limitations under the License.                                              #
###############################################################################

import math

import torch
from gpytorch.constraints.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from botorch.models import SingleTaskGP
from botorch.optim.fit import fit_gpytorch_torch
from volatilitygp.models import SingleTaskVariationalGP
from botorch.models.transforms import Standardize

from gpytorch.priors import Prior, GammaPrior
from torch.distributions import HalfCauchy
from torch.nn import Module as TModule


class HierarchicalHalfCauchyPrior(Prior, HalfCauchy):
    def __init__(self, scale, validate_args=None, transform=None):
        r""" 
        Hierarchical half cauchy priors as in https://arxiv.org/pdf/2103.00349.pdf
        """

        TModule.__init__(self)
        HalfCauchy.__init__(self, scale=scale, validate_args=validate_args)
        self._transform = transform

    def log_prob(self, input_tuple):
        lengthscales, h_scale = input_tuple
        lp1 = HalfCauchy(h_scale).log_prob(lengthscales).sum(-1)
        lp2 = HalfCauchy.log_prob(self, h_scale)
        return lp1 + lp2.to(lp1)


def _prepare_covar_module(
    ard_dims, lengthscale_constraint, outputscale_constraint, hyper=1.0, saas=True
):
    covar_module = MaternKernel(
        ard_num_dims=ard_dims, lengthscale_constraint=lengthscale_constraint, nu=2.5
    )
    if saas:
        # sparse axis aligned subspace priors for bo
        # eriksonn & jankowiak, uai '21
        covar_module.register_parameter(
            "raw_global_scale", torch.nn.Parameter(torch.zeros(1, requires_grad=True))
        )
        covar_module.register_prior(
            "lengthscale_prior",
            HierarchicalHalfCauchyPrior(hyper),
            lambda m: (m.lengthscale.reciprocal(), m.raw_global_scale.exp()),
        )
    covar_module = ScaleKernel(
        covar_module,
        outputscale_constraint=outputscale_constraint,
        outputscale_prior=GammaPrior(2.0, 0.15),
    )
    return covar_module


# GP Model
class GP(SingleTaskGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims,
        hyper=1.0,
        saas=True,
    ):
        covar_module = _prepare_covar_module(
            ard_dims,
            lengthscale_constraint,
            outputscale_constraint,
            hyper=hyper,
            saas=saas,
        )
        super(GP, self).__init__(
            train_x,
            train_y.view(-1, 1),
            likelihood,
            outcome_transform=Standardize(1),
            covar_module=covar_module,
        )
        self.ard_dims = ard_dims

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class VariationalGP(SingleTaskVariationalGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        lengthscale_constraint,
        outputscale_constraint,
        ard_dims,
        hyper=1.0,
        num_inducing=500,
    ):
        covar_module = _prepare_covar_module(
            ard_dims, lengthscale_constraint, outputscale_constraint, hyper=hyper
        )
        super(VariationalGP, self).__init__(
            init_points=train_x,
            likelihood=likelihood,
            covar_module=covar_module,
            num_inducing=num_inducing,
            init_targets=train_y.view(-1, 1),
            outcome_transform=Standardize(1),
        )
        self.ard_dims = ard_dims


def train_gp(
    train_x, train_y, use_ard, num_steps, hypers={}, method="exact", ls_hyper=1.0
):
    """Fit a GP model where train_x is in [0, 1]^d and train_y is standardized."""
    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]
    print("mem allocated: ", torch.cuda.memory_allocated() / (1024 ** 3))
    torch.cuda.empty_cache()

    # Create hyper parameter bounds
    noise_constraint = Interval(5e-4, 0.2)
    if use_ard:
        lengthscale_constraint = Interval(0.005, 2.0)
    else:
        lengthscale_constraint = Interval(
            0.005, math.sqrt(train_x.shape[1])
        )  # [0.005, sqrt(dim)]
    outputscale_constraint = Interval(0.05, 20.0)

    # Create models
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_y.dtype
    )
    ard_dims = train_x.shape[1] if use_ard else None

    base_cls = GP if method == "exact" else VariationalGP

    model = base_cls(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        lengthscale_constraint=lengthscale_constraint,
        outputscale_constraint=outputscale_constraint,
        ard_dims=ard_dims,
        hyper=ls_hyper,
    ).to(device=train_x.device, dtype=train_x.dtype)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    if method == "exact":
        mll = ExactMarginalLogLikelihood(likelihood, model)
    else:
        from gpytorch.mlls import PredictiveLogLikelihood

        mll = PredictiveLogLikelihood(likelihood, model, num_data=train_x.shape[-2])

    # Initialize model hypers
    if hypers:
        model.load_state_dict(hypers)
    else:
        model.covar_module.outputscale = 1.0
        model.covar_module.base_kernel.lengthscale = 0.5
        model.likelihood.noise = 0.005
    #         hypers = {}
    #         hypers["covar_module.outputscale"] = 1.0
    #         hypers["covar_module.base_kernel.lengthscale"] = 0.5
    #         hypers["likelihood.noise"] = 0.005
    #         model.initialize(**hypers)

    fit_gpytorch_torch(mll, options={"maxiter": num_steps, "lr": 0.01})
    model.eval()
    likelihood.eval()

    return model
