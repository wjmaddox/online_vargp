import torch

from unittest import TestCase, main

from gpytorch.lazy import DiagLazyTensor
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

from volatilitygp.models import SingleTaskVariationalGP
from volatilitygp.likelihoods import VolatilityGaussianLikelihood


class TestVariationalGPFantasyModel(TestCase):
    def setUp(self, ndata=20, seed=10):
        loc = torch.rand(ndata)
        scale = DiagLazyTensor(torch.rand(ndata))
        self.base_dist = MultivariateNormal(loc, scale)

    def test_nonbatch(self):
        rand_y = torch.randn_like(self.base_dist.mean)
        likelihood = VolatilityGaussianLikelihood()
        res = likelihood.expected_log_prob(rand_y, self.base_dist)
        self.assertEqual(res.shape, torch.Size((20,)))

    def test_multitask_nonbatch(self):
        base_dist = MultitaskMultivariateNormal(
            self.base_dist.mean.reshape(5, 4), self.base_dist.lazy_covariance_matrix
        )
        rand_y = torch.randn(5, 4)
        likelihood = VolatilityGaussianLikelihood(K=2, batch_shape=torch.Size((4,)))
        res = likelihood.expected_log_prob(rand_y, base_dist)
        self.assertEqual(res.shape, torch.Size((5,)))


if __name__ == "__main__":
    main()
