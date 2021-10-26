import torch

import torch.distributions as base_distributions

from gpytorch.likelihoods import GaussianLikelihood, Likelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.utils.pivoted_cholesky import pivoted_cholesky
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.constraints import Positive
from gpytorch.utils.errors import NotPSDError


class MultivariateNormalLikelihood(GaussianLikelihood):
    def __init__(
        self,
        num_train: int,
        rank: int = 1,
        batch_shape=torch.Size(),
        noise_covar_prior=None,
        noise_prior=None,
        noise_constraint=None,
    ):
        Likelihood.__init__(self)
        self.num_train = num_train

        self.register_parameter(
            name="noise_covar_factor",
            parameter=torch.nn.Parameter(torch.randn(*batch_shape, num_train, rank)),
        )
        if noise_covar_prior is not None:
            self.register_prior(
                "ErrorCovariancePrior", noise_covar_prior, lambda m: m._eval_covar_matrix
            )

        self.register_parameter(
            name="raw_noise",
            parameter=torch.nn.Parameter(torch.zeros(*batch_shape, num_train)),
        )
        if noise_constraint is None:
            noise_constraint = Positive()
        self.register_constraint("raw_noise", noise_constraint)
        if noise_prior is not None:
            self.register_prior("raw_noise_prior", noise_prior, lambda m: m.noise)

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    def _set_noise(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_noise)
        self.initialize(raw_noise=self.raw_noise_constraint.inverse_transform(value))

    @noise.setter
    def noise(self, value):
        self._set_noise(value)

    @property
    def noise_covar(self):
        if self.rank > 0:
            return self.noise_covar_factor.matmul(
                self.task_noise_covar_factor.transpose(-1, -2)
            )
        else:
            raise AttributeError(
                "Cannot retrieve task noises when covariance is diagonal."
            )

    @noise_covar.setter
    def noise_covar(self, value):
        # internally uses a pivoted cholesky decomposition to construct a low rank
        # approximation of the covariance
        if self.rank > 0:
            self.noise_covar_factor.data = pivoted_cholesky(value, max_iter=self.rank)
        else:
            raise AttributeError(
                "Cannot set non-diagonal task noises when covariance is diagonal."
            )

    def _eval_covar_matrix(self):
        covar_factor = self.noise_covar_factor
        noise = self.noise.unsqueeze(-1)
        D = noise * torch.eye(self.num_train, dtype=noise.dtype, device=noise.device)
        return covar_factor.matmul(covar_factor.transpose(-1, -2)) + D

    def marginal(self, function_dist, *params, **kwargs):
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        if self.training and mean.shape[-1] == self.num_train:
            covar = covar + self._eval_covar_matrix()
        else:
            if covar.shape[:-1] == self.noise.shape:
                covar = covar.add_diag(self.noise)
            elif covar.shape[:-1] == self.noise.shape[1:]:
                covar = covar.evaluate() + torch.diag_embed(self.noise)
            else:
                covar = covar.add_diag(self.noise[..., 0].unsqueeze(-1))

        return function_dist.__class__(mean, covar)

    def forward(self, function_samples, *params, **kwargs):
        if self.training and function_samples.shape[-1] == self.num_train:
            return MultivariateNormal(function_samples, self._eval_covar_matrix())
        else:
            noise = self.noise.view(*self.noise.shape[:-1], *function_samples.shape[-2:])
            return base_distributions.Independent(
                base_distributions.Normal(function_samples, noise.sqrt()), 1
            )


class FixedCovarMultivariateNormalLikelihood(MultivariateNormalLikelihood):
    def __init__(self, diag_term, covar_term, *args, **kwargs):
        super().__init__(
            num_train=covar_term.shape[-1],
            rank=covar_term.shape[-1],
            batch_shape=covar_term.shape[:-2],
            *args,
            **kwargs,
        )
        del self.noise_covar_factor
        try:
            self.noise_covar_factor = psd_safe_cholesky(covar_term, jitter=1e-3)
        except NotPSDError:
            evals, evecs = covar_term.symeig(eigenvectors=True)
            evals[evals < 0.0] = 0.0
            self.noise_covar_factor = psd_safe_cholesky(
                evecs.matmul(torch.diag_embed(evals)).matmul(evecs.transpose(-1, -2)),
                jitter=1e-3,
            )
        self.noise = diag_term.squeeze(-1).expand_as(self.noise)

    def get_fantasy_likelihood(self, *args, **kwargs):
        return super().get_fantasy_likelihood(*args, **kwargs)


class GeneralFixedCovarMultivariateNormalLikelihood(
    FixedCovarMultivariateNormalLikelihood
):
    def __init__(
        self,
        diag_term,
        covar_term,
        base_likelihood_class,
        likelihood_kwargs={},
        *args,
        **kwargs
    ):
        super().__init__(diag_term=diag_term, covar_term=covar_term, *args, **kwargs)
        self.eval_likelihood = base_likelihood_class(**likelihood_kwargs)

    def forward(self, function_samples, *params, **kwargs):
        if not self.training:
            return self.eval_likelihood.forward(function_samples, *params, **kwargs)
        else:
            return super().forward(function_samples, *params, **kwargs)
