from gpytorch.distributions import base_distributions
from gpytorch.lazy import DiagLazyTensor
from ._one_dimensional_likelihood import NewtonOneDimensionalLikelihood


class BernoulliLikelihood(NewtonOneDimensionalLikelihood):
    r"""
    Implements the Bernoulli likelihood used for GPs. Note that we use the natural parameterization,
    NOT the \phi(f) parameterization.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, function_samples, nobs=None, **kwargs):
        # we parameterize the logit as that's the natural parameterizatio for the distribution
        return base_distributions.Bernoulli(logits=function_samples)

    def grad_f(self, f, targets):
        # from wikipedia
        exp_f = f.exp()
        return (targets - (exp_f / (1 + exp_f))).unsqueeze(-1)

    def neg_hessian_f(self, f, targets=None):
        # second derivative of A(\eta)
        exp_f = f.exp()
        prob_f = exp_f / (1 + exp_f)
        return DiagLazyTensor(prob_f * (1 + prob_f))
