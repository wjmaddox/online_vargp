from gpytorch.lazy import DiagLazyTensor
from torch.distributions import (
    Poisson,
    RelaxedOneHotCategorical,
    RelaxedBernoulli,
)
from torch import Size

from ._one_dimensional_likelihood import NewtonOneDimensionalLikelihood


class _Poisson(Poisson):
    has_rsample = True

    def rsample(
        self, sample_shape=Size(), base_samples=None, cat_temp=0.001, bern_temp=1.0
    ):
        # temp hypers sort of tuned randomly

        max_rate = self.rate.max()
        # mean + 5 standard deviations should cover
        gumbel_range = torch.arange(
            0, int(max_rate * 6 + 1), device=self.rate.device, dtype=self.rate.dtype
        )

        # we compute the log probs out to 5sds and truncate, then compute a gumbel softmax prob
        pois_logprobs = self.log_prob(gumbel_range)
        softmax_dist = RelaxedOneHotCategorical(
            temperature=cat_temp, logits=pois_logprobs
        )
        # next, we compute relaxed bernoullis
        bern_dist = RelaxedBernoulli(
            temperature=bern_temp, probs=softmax_dist.rsample(sample_shape)
        )
        bern_samples = bern_dist.rsample()
        # and sum over the weights
        return (bern_samples * gumbel_range).sum(-1)


class PoissonLikelihood(NewtonOneDimensionalLikelihood):
    def grad_f(self, f, targets):
        return (targets - f.clamp(max=10.0).exp()).unsqueeze(-1)

    def neg_hessian_f(self, f, *args, **kwargs):
        return DiagLazyTensor(f.clamp(max=10.0).exp())

    def forward(self, function_samples, *args, **kwargs):
        return _Poisson(rate=function_samples.exp())
