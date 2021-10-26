from gpytorch.distributions import base_distributions
from gpytorch.lazy import DiagLazyTensor
from ._one_dimensional_likelihood import NewtonOneDimensionalLikelihood


class BinomialLikelihood(NewtonOneDimensionalLikelihood):
    r"""
    Implements the Binomial likelihood used for GPs. 
    """

    def __init__(self, nobs=100, *args, **kwargs):
        self.nobs = nobs
        super().__init__(*args, **kwargs)

    def forward(self, function_samples, nobs=None, **kwargs):
        # we parameterize the logit as that's the natural parameterizatio for the distribution
        return base_distributions.Binomial(
            total_count=self.nobs if nobs is None else nobs, logits=function_samples
        )

    def grad_f(self, f, targets):
        # from wikipedia
        exp_f = f.exp()
        return (targets - self.nobs * (exp_f / (1 + exp_f))).unsqueeze(-1)

    def neg_hessian_f(self, f, targets=None):
        # second derivative of A(\eta)
        exp_f = f.exp()
        return DiagLazyTensor(self.nobs * (exp_f / (1 + exp_f) ** 2))
