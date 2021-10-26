import torch

from unittest import TestCase, main

from gpytorch import lazify
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import VariationalELBO

from botorch.models import SingleTaskGP
from volatilitygp.models import SingleTaskVariationalGP
from volatilitygp.likelihoods import PoissonLikelihood


class TestVariationalGPFantasyModel(TestCase):
    def setUp(self, ndata=20, seed=10, likelihood="gaussian"):
        torch.set_default_dtype(torch.double)
        torch.random.manual_seed(seed)

        train_x = torch.randn(ndata, 1)
        train_y = torch.sin(4.0 * train_x)
        if likelihood != "gaussian":
            train_y = torch.distributions.Poisson(train_y.exp()).sample()

        inducing_points = torch.randn(int(ndata / 2), 1)
        self.model = SingleTaskVariationalGP(
            init_points=inducing_points,
            use_piv_chol_init=False,
            use_whitened_var_strat=True,
            train_inputs=train_x,
            train_targets=train_y,
            likelihood=PoissonLikelihood() if likelihood != "gaussian" else None,
        )

        if likelihood == "gaussian":
            self.model.likelihood.noise = 0.1

        # set the variational parameters to their optima
        with torch.no_grad():
            kernel_cross = self.model.covar_module(inducing_points, train_x)
            kernel_ind = self.model.covar_module(inducing_points)

            if likelihood == "gaussian":
                sigma_inv_update = (
                    kernel_cross.matmul(kernel_cross.transpose(-1, -2))
                    / self.model.likelihood.noise
                )
                rhs = train_y / self.model.likelihood.noise
            else:
                f = self.model.likelihood.newton_iteration(
                    train_x, train_y.view(-1), covar=self.model.covar_module(train_x)
                )
                hessian_inverse = self.model.likelihood.expected_hessian(f).inverse()
                sigma_inv_update = kernel_cross.matmul(hessian_inverse).matmul(
                    kernel_cross.transpose(-1, -2)
                )
                rhs = hessian_inverse.matmul(f)

            sigma_inv = kernel_ind + sigma_inv_update

            mean_cache = sigma_inv.inv_matmul(kernel_cross.matmul(rhs))
            var_mean = kernel_ind.matmul(mean_cache)

            var_cov = kernel_ind.matmul(sigma_inv.inv_matmul(kernel_ind.evaluate()))

            var_mean = var_mean.detach()
            var_cov_chol = var_cov.cholesky().detach()

        self.model.variational_strategy._variational_distribution.variational_mean.data = (
            var_mean
        )
        self.model.variational_strategy._variational_distribution.chol_variational_covar.data = (
            var_cov_chol
        )

        self.kernel_ind = kernel_ind

    def _test_nonbatch_fantasy(self, gaussian=True):
        fant_x = torch.randn(5, 1, requires_grad=True)
        fant_y = torch.sin(4.0 * fant_x).detach()  # .squeeze(-1)
        if not gaussian:
            fant_y = fant_y.exp().round()
        fantasy_model = self.model.condition_on_observations(fant_x, fant_y)
        self.assertIsInstance(fantasy_model, SingleTaskGP)

        ## now we check that we can backpropagate
        test_points = torch.linspace(-3, 3, 100)
        fantasy_model.eval()
        fant_predictive = fantasy_model(test_points)
        pred_samples = fant_predictive.rsample(torch.Size((64,)))
        pred_samples.norm().backward()
        self.assertIsNotNone(fant_x.grad)

    def _test_batch_fantasy(self, gaussian=True):
        fant_x = torch.randn(5, 1, requires_grad=True)
        fant_y = torch.sin(4.0 * fant_x.data) + torch.randn(64, 5, 1)
        if not gaussian:
            fant_y = fant_y.exp().round()
        # just quickly check that we can batched fantasize
        fantasy_model = self.model.get_fantasy_model(fant_x, fant_y)
        self.assertIsInstance(fantasy_model, SingleTaskGP)

        self.assertEqual(fantasy_model.train_targets.shape[:-1], torch.Size((64,)))

        fantasy_model.eval()
        test_points = torch.linspace(-3, 3, 12)
        pred_dist = fantasy_model(test_points)
        self.assertEqual(pred_dist.mean.shape, torch.Size((64, 12)))
        self.assertEqual(pred_dist.covariance_matrix.shape, torch.Size((64, 12, 12)))

        ## now we check that we can backpropagate
        test_points = torch.linspace(-3, 3, 100)
        fantasy_model.eval()
        fant_predictive = fantasy_model(test_points)
        pred_samples = fant_predictive.rsample(torch.Size((64,)))
        pred_samples.norm().backward()
        self.assertIsNotNone(fant_x.grad)

    def _test_fantasize(self, gaussian=True):
        fant_x = torch.randn(5, 1, requires_grad=True)
        fant_y = torch.sin(4.0 * fant_x).detach()
        if not gaussian:
            fant_y = fant_y.exp().round()
        fantasy_model = self.model.condition_on_observations(fant_x, fant_y)
        self.assertIsInstance(fantasy_model, SingleTaskGP)

    def test_nonbatch_fantasy_gaussian(self):
        self.setUp(likelihood="gaussian")
        self._test_nonbatch_fantasy(gaussian=True)

    def test_fantasize_gaussian(self):
        self.setUp(likelihood="gaussian")
        self._test_fantasize(gaussian=True)

    def test_batch_fantasy_gaussian(self):
        self.setUp(likelihood="gaussian")
        self._test_batch_fantasy(gaussian=True)

    def test_nonbatch_fantasy_poisson(self):
        self.setUp(likelihood="poisson")
        self._test_nonbatch_fantasy(gaussian=False)

    def test_fantasize_poisson(self):
        self.setUp(likelihood="poisson")
        self._test_fantasize(gaussian=False)

    def test_batch_fantasy_poisson(self):
        self.setUp(likelihood="poisson")
        self._test_batch_fantasy(gaussian=False)


if __name__ == "__main__":
    main()
