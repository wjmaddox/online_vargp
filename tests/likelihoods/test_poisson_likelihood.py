import torch

from unittest import TestCase, main

from gpytorch.lazy import DiagLazyTensor
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal

from volatilitygp.models import SingleTaskVariationalGP
from volatilitygp.likelihoods import PoissonLikelihood


class TestPoissonLikelihood(TestCase):
    def setUp(self, ndata=20, seed=10):
        loc = torch.rand(ndata)
        scale = DiagLazyTensor(torch.rand(ndata))
        self.base_dist = MultivariateNormal(loc, scale)

    def test_nonbatch(self):
        rand_y = torch.randn_like(self.base_dist.mean).exp().round()
        likelihood = PoissonLikelihood()
        res = likelihood.expected_log_prob(rand_y, self.base_dist)
        self.assertEqual(res.shape, torch.Size((20,)))

    def test_hessian(self):
        likelihood = PoissonLikelihood()
        # need the exact sample here
        response = likelihood(self.base_dist).sample()

        # hack to double check
        likelihood.f_stored = self.base_dist.mean
        likelihood.targets = response
        neg_hessian = likelihood.expected_hessian().evaluate()

        autograd_neg_hessian = super(PoissonLikelihood, likelihood).neg_hessian_f(
            f=likelihood.f_stored, targets=response
        )

        self.assertLessEqual((neg_hessian - autograd_neg_hessian).norm(), 1e-3)

    def test_gradient(self):
        likelihood = PoissonLikelihood()
        # need the exact sample here
        response = likelihood(self.base_dist).sample()

        grad = likelihood.grad_f(self.base_dist.mean, response)
        autograd_grad = super(PoissonLikelihood, likelihood).grad_f(
            self.base_dist.mean, response
        )
        self.assertLessEqual((grad - autograd_grad).norm(), 1e-3)
