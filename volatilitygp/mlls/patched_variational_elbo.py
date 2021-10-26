import torch

from gpytorch.mlls import VariationalELBO
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood


class PatchedApproximateMarginalLogLikelihood(_ApproximateMarginalLogLikelihood):
    """
    TODO: put this fix in gpytorch
    """

    def forward(self, approximate_dist_f, target, *params, **kwargs):
        r"""
        Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        Calling this function will call the likelihood's `expected_log_prob` function.

        Args:
            :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
                :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
            :attr:`target` (`torch.Tensor`):
                :math:`\mathbf y` The target values
            :attr:`**kwargs`:
                Additional arguments passed to the likelihood's `expected_log_prob` function.
        """
        # Get likelihood term and KL term
        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(
            approximate_dist_f, target, *params, **kwargs
        ).div(num_batch)
        kl_divergence = self.model.variational_strategy.kl_divergence().div(
            self.num_data / self.beta
        )

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for _, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior


class PatchedVariationalELBO(PatchedApproximateMarginalLogLikelihood, VariationalELBO):
    """
    TODO: put this fix in gytorch
    """

    def _log_likelihood_term(self, variational_dist_f, target, *params, **kwargs):
        return self.likelihood.expected_log_prob(
            target, variational_dist_f, *params, **kwargs
        ).sum(-1)
