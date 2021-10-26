from copy import deepcopy

import gpytorch
import torch

from botorch.models.model import Model
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from gpytorch.settings import cholesky_jitter
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.likelihoods import GaussianLikelihood

from ..mlls import StreamingAddedLossTerm

class VariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(
        self,
        inducing_points,
        mean_module=None,
        covar_module=None,
        streaming=False,
        likelihood=None,
        feat_extractor=None,
        beta=1.0,
        learn_inducing_locations=True,
    ):
        # the streaming option is to point towards an implementation of SSGPs
        # https://arxiv.org/abs/1705.07131
        # which stores copies of the old inducing prior & old inducing variational dist
        # we implement a GVI version because O-SVGP can have poor performance with small batches.
        data_dim = -2 if inducing_points.dim() > 1 else -1
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(data_dim)
        )

        variational_strategy = gpytorch.variational.UnwhitenedVariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)
        if mean_module is None:
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            self.mean_module = mean_module

        if covar_module is None:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        else:
            self.covar_module = covar_module

        self.streaming = streaming

        if self.streaming:
            self.register_added_loss_term("streaming_loss_term")
            self.old_variational_dist = None
            self.old_prior_dist = None
            self.old_inducing_points = None

        self.beta = beta
        if likelihood is not None:
            self.likelihood = likelihood
        else:
            self.likelihood = GaussianLikelihood()

        self.feat_extractor = feat_extractor
        self.train_checkpoint = deepcopy(self.state_dict())
        self.eval_checkpoint = deepcopy(self.state_dict())

    def __call__(self, inputs):
        if hasattr(self.variational_strategy, "_memoize_cache"):
            if (
                "cholesky_factor" in self.variational_strategy._memoize_cache
                and self.variational_strategy._memoize_cache["cholesky_factor"].shape[0]
                != inputs.shape[0]
            ):
                self.variational_strategy._memoize_cache.pop("cholesky_factor")
        if self.feat_extractor:
            inputs = self.feat_extractor(inputs)

        output_dist = super().__call__(inputs)
        if self.streaming and self.training:
            self.add_streaming_loss(inputs.shape[0], self.beta)

        return output_dist

    def add_streaming_loss(self, n, beta):
        if self.old_inducing_points is None:
            return None

        # first we create a new loss term
        # current_var_dist = self.variational_strategy.variational_distribution
        self.eval()
        current_var_dist = self(self.old_inducing_points)
        self.train()
        # self.training=True

        new_added_loss_term = StreamingAddedLossTerm(
            current_var_dist, self.old_variational_dist, self.old_prior_dist, beta / n,
        )
        self.update_added_loss_term("streaming_loss_term", new_added_loss_term)

    def register_streaming_loss(self):
        current_var_dist = self.variational_strategy.variational_distribution
        # then we update the distributional caches
        variational_covar = deepcopy(current_var_dist.covariance_matrix.detach())
        # TODO: try to remove the jitter?
        variational_covar = variational_covar + 1e-5 * torch.eye(
            variational_covar.shape[0],
            dtype=variational_covar.dtype,
            device=variational_covar.device,
        )
        self.old_variational_dist = gpytorch.distributions.MultivariateNormal(
            deepcopy(current_var_dist.mean.detach()).contiguous(), variational_covar.contiguous()
        )

        prior_dist = self.variational_strategy.prior_distribution
        self.old_prior_dist = gpytorch.distributions.MultivariateNormal(
            deepcopy(prior_dist.mean.detach()).contiguous(),
            deepcopy(prior_dist.covariance_matrix.detach()).contiguous(),
        )

        self.old_inducing_points = self.variational_strategy.inducing_points.clone().detach()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def param_groups(self, base_lr, learn_features=True):
        groups = []
        for name, param in self.named_parameters():
            if "raw" in name:
                groups.append({"params": param, "lr": base_lr})
            else:
                lr = base_lr / 10
                if "feat_extractor" in name and learn_features is False:
                    lr = 0.0
                groups.append({"params": param, "lr": lr})
        return groups

    def reset_checkpoints(self):
        self.train_checkpoint = deepcopy(self.state_dict())
        self.eval_checkpoint = deepcopy(self.state_dict())

    def set_streaming(self, streaming_state: bool):
        self.streaming = streaming_state
        self.register_added_loss_term("streaming_loss_term")
        self.old_variational_dist = None
        self.old_prior_dist = None
        if streaming_state is True:
            self.old_inducing_points = self.variational_strategy.inducing_points.clone().detach()
        else:
            self.old_inducing_points = None

    def update_variational_parameters(self, new_x, new_y, new_inducing_points=None):
        # if new_inducing_points = None
        # this version of the variational update does NOT assume that the inducing points
        # are moving around as we add new data. because we are using gradient based updates
        # to optimize them, we do not compute any type of randomized updates to them as in
        # bui et al.
        if new_inducing_points is None:
            new_inducing_points = self.variational_strategy.inducing_points.detach().clone()

        self.set_streaming(True)
        
        with torch.no_grad():
            self.register_streaming_loss()

            if len(new_y.shape) == 1:
                new_y = new_y.view(-1,1)

            S_a = self.variational_strategy.variational_distribution.lazy_covariance_matrix
            K_aa_old = self.variational_strategy.prior_distribution.lazy_covariance_matrix
            m_a = self.variational_strategy.variational_distribution.mean

            D_a_inv = (S_a.evaluate().inverse() - K_aa_old.evaluate().inverse())
            # compute D S_a^{-1} m_a
            pseudo_points = torch.solve(S_a.inv_matmul(m_a).unsqueeze(-1), D_a_inv)[0]
            # stack y and the pseudo points
            hat_y = torch.cat((new_y.view(-1,1), pseudo_points))

            # we now create Sigma_\hat y = blockdiag(\sigma^2 I; D_a)
            noise_diag = self.likelihood.noise * torch.eye(
                new_y.size(-2), device=new_y.device, dtype=new_y.dtype
            )
            zero_part = torch.zeros(
                new_y.size(-2), 
                pseudo_points.size(0), 
                dtype=new_y.dtype, 
                device=new_y.device
            )
            tophalf = torch.cat((noise_diag, zero_part), -1)
            bottomhalf = torch.cat((zero_part.t(), D_a_inv.inverse()),-1)
            sigma_hat_y = torch.cat((tophalf, bottomhalf))

            # stack the data to be able to compute covariances with it
            # (x, a)
            stacked_data = torch.cat((new_x, self.variational_strategy.inducing_points))
            
            K_fb = self.covar_module(stacked_data, new_inducing_points)
            K_bb = self.covar_module(new_inducing_points)
            # C = K_{hat f b} K_{bb}^{-1} K_{b hat f} + Sigma_\hat y
            pred_cov = K_fb @ (K_bb.inv_matmul(K_fb.evaluate().t())) + sigma_hat_y

            # the new mean is K_{hat f b} C^{-1} \hat y
            new_mean = K_fb.t() @ torch.solve(hat_y, pred_cov)[0].squeeze(-1).detach().contiguous()

            # the new covariance is K_bb - K_{hat f b} C^{-1} K_{b hat f}
            new_cov = K_bb - K_fb.t() @ torch.solve(K_fb.evaluate(), pred_cov)[0]
            new_variational_chol = psd_safe_cholesky(new_cov.evaluate(), jitter=cholesky_jitter.value()).detach().contiguous()
            self.variational_strategy._variational_distribution.variational_mean.data.mul_(0.).add_(new_mean)
            self.variational_strategy._variational_distribution.chol_variational_covar.data.mul_(0.).add_(new_variational_chol)
            self.variational_strategy.inducing_points.data.mul_(0.).add_(new_inducing_points.detach())

    def _update_variational_covariance(self, test_point, noise_term, is_whitened=True):
        # S matrix
        # var_cov = self.variational_strategy.variational_distribution.lazy_covariance_matrix
        var_cov_root = gpytorch.lazify(self.variational_strategy._variational_distribution.chol_variational_covar)
        var_cov = var_cov_root.matmul(var_cov_root.transpose(-1, -2))
        
        # K_{*m}
        test_inducing_kernel = self.covar_module(test_point, self.variational_strategy.inducing_points).evaluate_kernel()

        # K^{1/2}
        try:
            Kmm_root = self.variational_strategy._memoize_cache["cholesky_factor"]
            has_kmm = False
        except:
            Kmm = self.covar_module(self.variational_strategy.inducing_points)
            Kmm_root = Kmm.cholesky()
            has_kmm = True

        if is_whitened:
            # K^{-1/2} S^{1/2}
            Kmm_root_s = Kmm_root.inv_matmul(var_cov_root.evaluate())#.float()       

            inner_matrix_root = test_inducing_kernel.matmul(Kmm_root_s)
            inner_matrix = gpytorch.lazify(
                inner_matrix_root @ inner_matrix_root.transpose(-1, -2)
            ).add_diag(self.likelihood.noise)
            
            proj_root = test_inducing_kernel.matmul(Kmm_root_s)
            delta_s = proj_root.transpose(-1, -2).matmul(inner_matrix.inv_matmul(proj_root))
        else:
            # we need to get out K_mm
            if not has_kmm:
                Kmm = Kmm_root.matmul(Kmm_root.transpose(-1, -2))

            proj_solve = Kmm.inv_matmul(test_inducing_kernel.transpose(-1, -2).evaluate())
            
            inner_matrix_root = var_cov_root.transpose(-1, -2).matmul(proj_solve)
            inner_matrix = gpytorch.lazify(
                inner_matrix_root.transpose(-1, -2).matmul(inner_matrix_root)
            ).add_diag(self.likelihood.noise)
            inner_mat_as_inducing = proj_solve.matmul(inner_matrix.inv_matmul(proj_solve.transpose(-1, -2)))
            delta_s = var_cov.matmul(gpytorch.lazify(inner_mat_as_inducing)).matmul(var_cov)

        res = var_cov - delta_s
        try:
            res_chol = res.cholesky()
        except gpytorch.utils.errors.NotPSDError:
            print("Resulting to chopping the non psd parts!")
            evals, evecs = res.symeig(eigenvectors = True)
            res_psd = evecs.matmul(torch.diag_embed(evals)).matmul(evecs.transpose(-1, -2).evaluate())
            res_chol = gpytorch.lazify(res_psd).cholesky()
        return gpytorch.lazy.CholLazyTensor(res_chol)
    
    def _update_variational_mean(self, test_point, test_response, new_covariance, is_whitened=True):
        # S matrix
        # var_cov = self.variational_strategy.variational_distribution.lazy_covariance_matrix
        var_cov_root = gpytorch.lazify(self.variational_strategy._variational_distribution.chol_variational_covar)
        var_cov = var_cov_root.matmul(var_cov_root.transpose(-1, -2))
        
        # K^{1/2}
        try:
            Kmm_root = self.variational_strategy._memoize_cache["cholesky_factor"]
        except:
            Kmm = self.covar_module(self.variational_strategy.inducing_points)
            Kmm_root = Kmm.cholesky()
    
        # K_{*m}
        test_inducing_kernel = self.covar_module(test_point, self.variational_strategy.inducing_points)
        
        old_term = var_cov.inv_matmul(self.variational_strategy.variational_distribution.mean)

        rhs = test_inducing_kernel.transpose(-1, -2).matmul(test_response) / self.likelihood.noise
        
        new_term = Kmm_root.inv_matmul(rhs)

        if not is_whitened:
            new_term = Kmm_root.transpose(-1, -2).inv_matmul(new_term)

        new_mean = old_term + new_term
        return new_covariance.matmul(new_mean).squeeze(-1)

    def get_fantasy_model(self, inputs, targets, noise_term=None, **kwargs):
        is_whitened = not isinstance(self.variational_strategy, gpytorch.variational.UnwhitenedVariationalStrategy)
        new_covariance = self._update_variational_covariance(inputs, noise_term, is_whitened=is_whitened)
        new_mean = self._update_variational_mean(inputs, targets, new_covariance, is_whitened=is_whitened)

        new_mean = new_mean.contiguous()
        new_chol_covariance = new_covariance.root_decomposition().root.evaluate()

        cloned_inducing_points = self.variational_strategy.inducing_points.clone().detach_()
        if self.variational_strategy.inducing_points.requires_grad:
            learn_inducing_locations = True
            cloned_inducing_points.requires_grad_()
        else:
            learn_inducing_locations = False

        new_model = VariationalGPModel(
            inducing_points=cloned_inducing_points,
            mean_module=self.mean_module,
            covar_module=self.covar_module,
            streaming=self.streaming,
            likelihood=self.likelihood,
            feat_extractor=self.feat_extractor,
            beta=self.beta,
            learn_inducing_locations=learn_inducing_locations,
        )

        # we need to delete the parameters to allow backprop wrt input
        # this means that we can't train with this model in the future
        del new_model.variational_strategy._variational_distribution.variational_mean
        new_model.variational_strategy._variational_distribution.variational_mean = new_mean
        del new_model.variational_strategy._variational_distribution.chol_variational_covar
        new_model.variational_strategy._variational_distribution.chol_variational_covar = new_chol_covariance
        new_model.variational_strategy.variational_params_initialized.fill_(1)
        return new_model

class ApproximateGPyTorchModel(Model):
    def __init__(self, model, likelihood, num_outputs, *args, **kwargs):
        super().__init__()
        
        self.model = model
        self.likelihood = likelihood
        self._desired_num_outputs = num_outputs

    @property
    def num_outputs(self):
        return self._desired_num_outputs

    def posterior(self, X, output_indices=None, observation_noise=False, *args, **kwargs):
        self.model.eval()
        self.likelihood.eval()
        dist = self.model(X)
        if observation_noise:
            dist = self.likelihood(dist, *args, **kwargs)
        return GPyTorchPosterior(mvn=dist)

    def forward(self, X, *args, **kwargs):
        return self.model(X)
