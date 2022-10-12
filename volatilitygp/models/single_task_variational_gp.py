from typing import Union
from copy import deepcopy

import torch
import functools

from botorch.models.gpytorch import GPyTorchModel
from botorch.models import SingleTaskGP
from botorch.posteriors import GPyTorchPosterior

from gpytorch import lazify
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import (
    CholLazyTensor,
    TriangularLazyTensor,
)
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood as FNGaussianLikelihood
from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.utils.errors import NotPSDError
from gpytorch.utils.memoize import cached, add_to_cache, clear_cache_hook
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)

from ..utils import pivoted_cholesky_init


def _update_caches(m, *args, **kwargs):
    if hasattr(m, "_memoize_cache"):
        for key, item in m._memoize_cache.items():
            if type(item) is not tuple and type(item) is not MultivariateNormal:
                if len(args) is 0:
                    new_lc = item.to(torch.empty(0, **kwargs))
                else:
                    new_lc = item.to(*args)
                m._memoize_cache[key] = new_lc
                if type(item) is TriangularLazyTensor:
                    m._memoize_cache[key] = m._memoize_cache[key].double()
            elif type(item) is MultivariateNormal:
                if len(args) is 0:
                    new_lc = item.lazy_covariance_matrix.to(torch.empty(0, **kwargs))
                else:
                    new_lc = item.lazy_covariance_matrix.to(*args)
                m._memoize_cache[key] = MultivariateNormal(
                    item.mean.to(*args, **kwargs), new_lc
                )
            else:
                m._memoize_cache[key] = (x.to(*args, **kwargs) for x in item)


def _add_cache_hook(tsr, pred_strat):
    if tsr.grad_fn is not None:
        wrapper = functools.partial(clear_cache_hook, pred_strat)
        functools.update_wrapper(wrapper, clear_cache_hook)
        tsr.grad_fn.register_hook(wrapper)
    return tsr


class _SingleTaskVariationalGP(ApproximateGP):
    def __init__(
        self,
        init_points=None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
    ):

        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel())

        if use_piv_chol_init:
            if num_inducing is None:
                num_inducing = int(init_points.shape[-2] / 2)

            if num_inducing < init_points.shape[-2]:
                covar_module = covar_module.to(init_points)

                covariance = covar_module(init_points)
                if init_targets is not None and init_targets.shape[-1] == 1:
                    init_targets = init_targets.squeeze(-1)
                if likelihood is not None and not isinstance(
                    likelihood, GaussianLikelihood
                ):
                    _ = likelihood.newton_iteration(
                        init_points, init_targets, model=None, covar=covariance
                    )
                    if likelihood.has_diag_hessian:
                        hessian_sqrt = likelihood.expected_hessian().sqrt()
                    else:
                        hessian_sqrt = (
                            lazify(likelihood.expected_hessian())
                            .root_decomposition()
                            .root
                        )
                    covariance = hessian_sqrt.matmul(covariance).matmul(
                        hessian_sqrt.transpose(-1, -2)
                    )
                inducing_points = pivoted_cholesky_init(
                    init_points, covariance.evaluate(), num_inducing
                )
            else:
                inducing_points = init_points.detach().clone()
        else:
            inducing_points = init_points.detach().clone()

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.shape[-2]
        )
        if use_whitened_var_strat:
            variational_strategy = VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        else:
            variational_strategy = UnwhitenedVariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=learn_inducing_locations,
            )
        super(_SingleTaskVariationalGP, self).__init__(variational_strategy)
        self.mean_module = ConstantMean() if mean_module is None else mean_module
        self.mean_module.to(init_points)
        self.covar_module = covar_module

        self.likelihood = GaussianLikelihood() if likelihood is None else likelihood
        self.likelihood.to(init_points)
        self.train_inputs = [train_inputs] if train_inputs is not None else [init_points]
        self.train_targets = train_targets if train_targets is not None else init_targets

        self.condition_into_exact = True

        self.to(init_points)

    @cached(name="pseudo_points")
    def pseudo_points(self):
        is_whitened = not isinstance(
            self.variational_strategy, UnwhitenedVariationalStrategy
        )

        # retrieve the variational mean, m and covariance matrix, S.
        var_cov_root = TriangularLazyTensor(
            self.variational_strategy._variational_distribution.chol_variational_covar
        )
        var_cov = CholLazyTensor(var_cov_root)
        var_mean = (
            self.variational_strategy.variational_distribution.mean
        )  # .unsqueeze(-1)
        if var_mean.shape[-1] != 1:
            var_mean = var_mean.unsqueeze(-1)

        if is_whitened:
            # compute R = I - S
            res = var_cov.add_jitter(-1.0)._mul_constant(-1.0)

            # K^{1/2}
            Kmm = self.covar_module(self.variational_strategy.inducing_points)
            Kmm_root = Kmm.cholesky()
        else:
            # R = K - S
            Kmm = self.covar_module(self.variational_strategy.inducing_points)
            res = Kmm - var_cov

        cov_diff = res

        # D_a = (S^{-1} - K^{-1})^{-1} = S + S R^{-1} S
        # note that in the whitened case R = I - S, unwhitened R = K - S
        # we compute (R R^{T})^{-1} R^T S for stability reasons as R is probably not PSD.
        eval_lhs = var_cov.evaluate()
        eval_rhs = cov_diff.transpose(-1, -2).matmul(eval_lhs)
        inner_term = cov_diff.matmul(cov_diff.transpose(-1, -2))
        inner_solve = inner_term.add_jitter(1e-3).inv_matmul(
            eval_rhs, eval_lhs.transpose(-1, -2)
        )
        inducing_covar = var_cov + inner_solve

        if is_whitened:
            inducing_covar = Kmm_root.matmul(inducing_covar).matmul(
                Kmm_root.transpose(-1, -2)
            )

        # mean term: D_a S^{-1} m
        # unwhitened: (S - S R^{-1} S) S^{-1} m = (I - S R^{-1}) m
        rhs = cov_diff.transpose(-1, -2).matmul(var_mean)
        inner_rhs_mean_solve = inner_term.add_jitter(1e-3).inv_matmul(rhs)
        if not is_whitened:
            new_mean = var_mean + var_cov.matmul(inner_rhs_mean_solve)
        else:
            new_mean = Kmm_root.matmul(inner_rhs_mean_solve)

        # ensure inducing covar is psd
        try:
            final_inducing_covar = CholLazyTensor(
                inducing_covar.add_jitter(1e-3).cholesky()
            ).evaluate()
        except NotPSDError:
            from gpytorch.lazy import DiagLazyTensor

            evals, evecs = inducing_covar.symeig(eigenvectors=True)
            final_inducing_covar = (
                evecs.matmul(DiagLazyTensor(evals + 1e-4))
                .matmul(evecs.transpose(-1, -2))
                .evaluate()
            )

        return final_inducing_covar, new_mean

    @cached(name="inducing_model")
    def inducing_model(self):
        with torch.no_grad():
            inducing_noise_covar, inducing_mean = self.pseudo_points()
            inducing_points = self.variational_strategy.inducing_points.detach()

            if hasattr(self, "input_transform"):
                [p.detach_() for p in self.input_transform.buffers()]

            new_covar_module = deepcopy(self.covar_module)
            if not self.condition_into_exact:
                new_covar_module = InducingPointKernel(
                    new_covar_module, inducing_points, self.likelihood
                )

            inducing_exact_model = SingleTaskGP(
                inducing_points,
                inducing_mean,
                covar_module=deepcopy(self.covar_module),
                input_transform=deepcopy(self.input_transform)
                if hasattr(self, "input_transform")
                else None,
                outcome_transform=deepcopy(self.outcome_transform)
                if hasattr(self, "outcome_transform")
                else None,
            )
            inducing_exact_model.mean_module = deepcopy(self.mean_module)
            inducing_exact_model.likelihood = deepcopy(self.likelihood)

            if isinstance(inducing_exact_model.likelihood, FNGaussianLikelihood):
                inducing_exact_model.likelihood.noise = (
                    self.likelihood.noise.mean().detach().expand(inducing_mean.shape[:-1])
                )

            # now fantasize around this model
            is_non_gaussian = not (
                isinstance(self.likelihood, GaussianLikelihood)
                or isinstance(self.likelihood, FNGaussianLikelihood)
            )

            # construct pseudo targets if likelihood is not Gaussian
            # this initializes the likelihood pseudo caches to enable computation of the hessian inverse
            if is_non_gaussian:
                _ = inducing_exact_model.likelihood.newton_iteration(
                    finit=torch.zeros_like(inducing_mean.squeeze(-1)),
                    inputs=inducing_points,
                    targets=self.likelihood(
                        self(inducing_points), return_gaussian=False
                    ).sample()[0],
                    covar=self.covar_module(inducing_points),
                )

            # as this model is new, we need to compute a posterior to construct the prediction strategy
            # which uses the likelihood pseudo caches
            faked_points = torch.randn(
                *inducing_points.shape[:-2],
                1,
                inducing_points.shape[-1],
                device=inducing_points.device,
                dtype=inducing_points.dtype,
            )
            _ = inducing_exact_model.posterior(faked_points)

            # then we overwrite the likelihood to take into account the multivariate normal term
            pred_strat = inducing_exact_model.prediction_strategy
            pred_strat._memoize_cache = {}
            with torch.no_grad():
                updated_lik_train_train_covar = (
                    pred_strat.train_prior_dist.lazy_covariance_matrix
                    + inducing_noise_covar
                )
                pred_strat.lik_train_train_covar = updated_lik_train_train_covar

            # do the mean cache because the mean cache doesn't solve against lik_train_train_covar
            train_mean = inducing_exact_model.mean_module(
                *inducing_exact_model.train_inputs
            )
            train_labels_offset = (
                inducing_exact_model.prediction_strategy.train_labels - train_mean
            ).unsqueeze(-1)
            mean_cache = updated_lik_train_train_covar.inv_matmul(
                train_labels_offset
            ).squeeze(-1)
            mean_cache = _add_cache_hook(
                mean_cache, inducing_exact_model.prediction_strategy
            )
            add_to_cache(pred_strat, "mean_cache", mean_cache)

            inducing_exact_model.prediction_strategy = pred_strat
        return inducing_exact_model

    def get_fantasy_model(
        self,
        inputs,
        targets,
        noise=None,
        condition_into_sgpr=False,
        targets_are_gaussian=True,
        **kwargs,
    ):
        #####################################
        # first we construct an exact model over the inducing points with the inducing covariance
        # matrix from SGPR
        #####################################

        is_non_gaussian = not (
            isinstance(self.likelihood, GaussianLikelihood)
            or isinstance(self.likelihood, FNGaussianLikelihood)
        )

        inducing_exact_model = self.inducing_model()

        # construct pseudo targets if likelihood is not Gaussian
        if is_non_gaussian and not targets_are_gaussian:
            orig_targets = targets
            targets = inducing_exact_model.likelihood.newton_iteration(
                inputs=inputs, targets=targets, covar=self.covar_module(inputs),
            )

        ###############################################
        # then we update this model by adding in the inputs and pseudo targets
        ###############################################

        if is_non_gaussian:
            inducing_exact_model.likelihood.f_stored = targets
            if not targets_are_gaussian:
                inducing_exact_model.likelihood.targets = orig_targets

        if inputs.shape[-2] == 1 or targets.shape[-1] != 1:
            targets = targets.unsqueeze(-1)
            # put on a trailing bdim for bs of 1
        # if inputs.shape[:-2] != targets.shape[1:-1]:
        #    targets = targets.expand(targets.shape[0], *inputs.shape[:-2], targets.shape[-1])
        # finally we fantasize wrt targets
        if noise is not None:
            if noise.shape[-1] != targets.shape[-1]:
                noise = noise.unsqueeze(-1)
            kwargs["noise"] = noise
        if is_non_gaussian and inputs.shape[:-1] != targets.shape:
            inputs = inputs.expand(*targets.shape[:-1], *inputs.shape[-1:])
        fantasy_model = inducing_exact_model.condition_on_observations(
            inputs, targets, **kwargs
        )
        fant_pred_strat = fantasy_model.prediction_strategy

        # first we update the lik_train_train_covar
        # do the mean cache again because the mean cache resets the likelihood forward
        train_mean = fantasy_model.mean_module(*fantasy_model.train_inputs)
        train_labels_offset = (fant_pred_strat.train_labels - train_mean).unsqueeze(-1)
        fantasy_lik_train_root_inv = (
            fant_pred_strat.lik_train_train_covar.root_inv_decomposition()
        )
        mean_cache = fantasy_lik_train_root_inv.matmul(train_labels_offset).squeeze(-1)
        mean_cache = _add_cache_hook(mean_cache, fant_pred_strat)
        add_to_cache(fant_pred_strat, "mean_cache", mean_cache)

        fantasy_model.prediction_strategy = fant_pred_strat
        return fantasy_model

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred

    def to(self, *args, **kwargs):
        _update_caches(self, *args, **kwargs)
        self.variational_strategy = self.variational_strategy.to(*args, **kwargs)
        _update_caches(self.variational_strategy, *args, **kwargs)
        return super().to(*args, **kwargs)


class SingleTaskVariationalGP(_SingleTaskVariationalGP, GPyTorchModel):
    def __init__(
        self,
        init_points=None,
        likelihood=None,
        learn_inducing_locations=True,
        covar_module=None,
        mean_module=None,
        use_piv_chol_init=True,
        num_inducing=None,
        use_whitened_var_strat=True,
        init_targets=None,
        train_inputs=None,
        train_targets=None,
        outcome_transform=None,
        input_transform=None,
    ):
        if outcome_transform is not None:
            is_gaussian_likelihood = (
                isinstance(likelihood, GaussianLikelihood) or likelihood is None
            )
            if train_targets is not None and is_gaussian_likelihood:
                if train_targets.ndim == 1:
                    train_targets = train_targets.unsqueeze(-1)
                train_targets, _ = outcome_transform(train_targets)

            if init_targets is not None and is_gaussian_likelihood:
                init_targets, _ = outcome_transform(init_targets)
                init_targets = init_targets.squeeze(-1)

        if train_targets is not None:
            train_targets = train_targets.squeeze(-1)

        # unlike in the exact gp case we need to use the input transform to pre-define the inducing pts
        if input_transform is not None:
            if init_points is not None:
                init_points = input_transform(init_points)

        _SingleTaskVariationalGP.__init__(
            self,
            init_points=init_points,
            likelihood=likelihood,
            learn_inducing_locations=learn_inducing_locations,
            covar_module=covar_module,
            mean_module=mean_module,
            use_piv_chol_init=use_piv_chol_init,
            num_inducing=num_inducing,
            use_whitened_var_strat=use_whitened_var_strat,
            init_targets=init_targets,
            train_inputs=train_inputs,
            train_targets=train_targets,
        )

        if input_transform is not None:
            self.input_transform = input_transform.to(
                self.variational_strategy.inducing_points
            )

        if outcome_transform is not None:
            self.outcome_transform = outcome_transform.to(
                self.variational_strategy.inducing_points
            )

    def forward(self, x):
        x = self.transform_inputs(x)
        return super().forward(x)

    @property
    def num_outputs(self) -> int:
        # we should only be able to have one output without a multitask variational strategy here
        return 1

    def posterior(
        self,
        X: torch.Tensor,
        observation_noise: Union[bool, torch.Tensor] = False,
        **kwargs,
    ):
        if observation_noise and not isinstance(self.likelihood, _GaussianLikelihoodBase):
            noiseless_posterior = super().posterior(
                X=X, observation_noise=False, **kwargs
            )
            noiseless_mvn = noiseless_posterior.mvn
            neg_hessian_f = self.likelihood.neg_hessian_f(noiseless_mvn.mean)
            try:
                likelihood_cov = neg_hessian_f.inverse()
            except:
                eye_like_hessian = torch.eye(
                    neg_hessian_f.shape[-2],
                    device=neg_hessian_f.device,
                    dtype=neg_hessian_f.dtype,
                )
                likelihood_cov = lazify(neg_hessian_f).inv_matmul(eye_like_hessian)

            noisy_mvn = type(noiseless_mvn)(
                noiseless_mvn.mean, noiseless_mvn.lazy_covariance_matrix + likelihood_cov
            )
            return GPyTorchPosterior(mvn=noisy_mvn)

        return super().posterior(X=X, observation_noise=observation_noise, **kwargs)


class FixedNoiseVariationalGP(SingleTaskVariationalGP):
    def __init__(self, init_y_var=None, *args, **kwargs):
        from volatilitygp.likelihoods import FixedNoiseGaussianLikelihood

        likelihood = FixedNoiseGaussianLikelihood(init_y_var)
        super().__init__(likelihood=likelihood, *args, **kwargs)

    def get_fantasy_model(
        self,
        inputs,
        targets,
        noise=None,
        condition_into_sgpr=False,
        targets_are_gaussian=True,
        **kwargs,
    ):
        if noise is None:
            noise = self.likelihood.noise.mean().expand(targets.shape[1:]).detach()
        return super().get_fantasy_model(
            inputs,
            targets,
            noise,
            condition_into_sgpr,
            targets_are_gaussian=True,
            **kwargs,
        )
