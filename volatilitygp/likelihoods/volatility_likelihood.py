import torch

from torch.distributions import Normal
from gpytorch.constraints import Positive, Interval

from ._one_dimensional_likelihood import NewtonOneDimensionalLikelihood


class VolatilityGaussianLikelihood(NewtonOneDimensionalLikelihood):
    def __init__(self, K=5, batch_shape=torch.Size(), *args, **kwargs):
        """
        parameterization of gaussian likelihood for volatility models like in 
        wilson & ghahramani, copula processes, eq. 21.
        """

        super().__init__()
        self.raw_a = torch.nn.Parameter(torch.rand(*batch_shape, K, requires_grad=True))
        raw_b_init = 0.1 * torch.rand(*batch_shape, K)
        self.raw_b = torch.nn.Parameter(raw_b_init.detach_().requires_grad_())
        self.raw_c = torch.nn.Parameter(torch.rand(*batch_shape, K, requires_grad=True))

        self.register_constraint("raw_a", Positive())
        self.register_constraint("raw_b", Interval(-3.0, 3.0))
        self.register_constraint("raw_c", Interval(-3.0, 3.0))

    @property
    def trans_a(self):
        return self.raw_a_constraint.transform(self.raw_a)

    @property
    def trans_b(self):
        return self.raw_b_constraint.transform(self.raw_b)

    @property
    def trans_c(self):
        return self.raw_c_constraint.transform(self.raw_c)

    def forward(self, function_samples, *args, **kwargs):
        transform = (
            (self.trans_b * function_samples.unsqueeze(-1) + self.trans_c).exp() + 1
        ).log() * self.trans_a
        summed_transform = transform.sum(-1)
        return Normal(torch.zeros_like(summed_transform), summed_transform + 1e-6)

    def expected_log_prob(self, target, input, *params, **kwargs):
        res = super().expected_log_prob(target, input, *params, **kwargs)
        num_event_dim = len(input.event_shape)
        if num_event_dim > 1:
            res = res.sum(-1)
        return res


# TODO: use a multitask Gaussian likelihood somehow in the multitask setting
