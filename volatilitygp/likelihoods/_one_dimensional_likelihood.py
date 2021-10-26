import torch
import copy

from gpytorch import lazify
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _OneDimensionalLikelihood, FixedNoiseGaussianLikelihood

from .multivariate_normal_likelihood import FixedCovarMultivariateNormalLikelihood


class NewtonOneDimensionalLikelihood(_OneDimensionalLikelihood):
    f_stored = None
    should_return_gaussian_on_eval = True
    has_diag_hessian = True

    def __call__(self, input, *args, **kwargs):
        return_gaussian = kwargs.pop("return_gaussian", True)

        if not self.training and isinstance(input, MultivariateNormal):
            if self.should_return_gaussian_on_eval and return_gaussian:
                targets = args[0]
                if isinstance(targets, list):
                    targets = targets[0].squeeze(-1)
                # noise_cov = self.expected_hessian(input.mean, targets).inverse()
                expected_hessian = self.expected_hessian(input.mean, targets)
                if not isinstance(expected_hessian, torch.Tensor):
                    noise_cov = expected_hessian.inverse()
                else:
                    eye_like_hessian = torch.eye(
                        expected_hessian.shape[-1],
                        device=expected_hessian.device,
                        dtype=expected_hessian.dtype,
                    )
                    noise_cov = torch.solve(
                        eye_like_hessian,
                        lazify(expected_hessian).add_jitter(1e-4).evaluate(),
                    )[0]
                    evals, evecs = noise_cov.symeig(eigenvectors=True)
                    evals[evals <= 0.0] = 0.0
                    noise_cov = evecs.matmul(torch.diag_embed(evals)).matmul(
                        evecs.transpose(-1, -2)
                    )

                return MultivariateNormal(
                    input.mean, input.lazy_covariance_matrix + noise_cov
                )
        return super().__call__(input, *args, **kwargs)

    def __deepcopy__(self, memo):
        if self.f_stored is None:
            return self
        else:
            old_f_stored = self.f_stored
            self.f_stored = None
            old_targets = self.targets
            self.targets = None
            new_likelihood = copy.deepcopy(self)
            self.f_stored = old_f_stored
            self.targets = old_targets
            return new_likelihood

    def newton_iteration(
        self,
        inputs,
        targets,
        model=None,
        finit=None,
        cutoff=2e-1,
        covar=None,
        maxiter=1e3,
    ):
        finit = (
            torch.zeros(
                *targets.shape[:-2],
                inputs.shape[-2],
                dtype=inputs.dtype,
                device=inputs.device
            )
            if finit is None
            else finit
        )

        fold = 10000.0 * torch.ones_like(finit)
        f = finit
        i = 0

        if not f.requires_grad:
            f = f.requires_grad_()
            f_as_leaf = True

        if model is not None:
            kmat = model.covar_module(inputs)
        else:
            kmat = covar

        has_passed = torch.zeros(finit.shape[:-2], device=finit.device).bool()

        while ((f - fold).norm().sum(-1) > cutoff).any():
            i += 1
            if f_as_leaf and f.grad is not None:
                f.grad.zero_()
            w = self.neg_hessian_f(f, targets=targets)

            # put in a hessian eval convergence check to prevent overshoot
            if not self.has_diag_hessian:
                if w.symeig()[0].norm() < cutoff:
                    f = fold
                    break

            if self.has_diag_hessian:
                if isinstance(w, torch.Tensor):
                    w = w.clamp(min=1e-6)
                w_sqrt = w.sqrt()
            else:
                try:
                    w_sqrt = lazify(w).cholesky().evaluate()
                except:
                    f = finit
                    break
            inner_mat = lazify(
                w_sqrt.transpose(-1, -2).matmul(kmat.matmul(w_sqrt))
            ).add_jitter(1.0)
            grad_f = self.grad_f(f, targets).reshape(*f.shape, -1)

            # put in a gradient convergence check
            if not self.has_diag_hessian:
                if grad_f.norm() < cutoff:
                    f = fold
                    break

            b = w.matmul(f.reshape(*f.shape, -1)) + grad_f
            updated_rhs = w_sqrt.transpose(-1, -2).matmul(kmat.matmul(b))
            update = w_sqrt.matmul(inner_mat.inv_matmul(updated_rhs))

            fold = f.clone()
            f = kmat.matmul(b - update).reshape(*f.shape)

            # we don't update the batch steps that have alreadtargets passed
            with torch.no_grad():
                elementwise_checks = (f - fold).norm().sum(-1) <= cutoff
                has_passed = (elementwise_checks + has_passed).bool()
                f[has_passed].data = fold[has_passed].data

            if i > maxiter:
                f = finit
                break

        self.f_stored = f
        self.targets = targets
        return f

    def expected_hessian(self, f=None, targets=None, *args, **kwargs):
        if f is None:
            return self.neg_hessian_f(self.f_stored, targets=self.targets)
        else:
            return self.neg_hessian_f(f, targets=targets)

    def grad_f(self, f, targets):
        forward_fn = lambda input: self.forward(input).log_prob(targets).sum(-1)
        with torch.enable_grad():
            return torch.autograd.functional.jacobian(
                forward_fn, f, create_graph=True
            ).unsqueeze(-1)

    def neg_hessian_f(self, f, targets=None):
        if targets.ndim > 1:
            hessian_stacked = []
            for batched_targets in targets:
                hessian_stacked.append(
                    NewtonOneDimensionalLikelihood.neg_hessian_f(
                        self, f=f, targets=batched_targets
                    ).unsqueeze(0)
                )
            return torch.cat(hessian_stacked, dim=0)
        else:
            forward_fn = (
                lambda input: self.forward(input.double())
                .log_prob(
                    targets.double() if not targets.dtype == torch.long else targets
                )
                .sum(-1)
                .to(input.dtype)
            )
            with torch.enable_grad():
                hessian_f = -1.0 * torch.autograd.functional.hessian(
                    forward_fn, f, create_graph=True
                )
            return hessian_f

    def get_fantasy_likelihood(self, *args, **kwargs):
        if self.has_diag_hessian:
            inv_hessian = self.expected_hessian().inverse().diag().clamp(min=0.0)
            return FixedNoiseGaussianLikelihood(
                noise=inv_hessian, batch_shape=inv_hessian.shape[:-1]
            )
        else:
            hessian = lazify(self.expected_hessian()).add_jitter(1e-6)
            eye_like_hessian = torch.eye(
                hessian.shape[-1], device=hessian.device, dtype=hessian.dtype
            )
            inv_hessian = hessian.inv_matmul(eye_like_hessian)
            return FixedCovarMultivariateNormalLikelihood(
                1e-6 * eye_like_hessian.diag().unsqueeze(-1), covar_term=inv_hessian
            )
