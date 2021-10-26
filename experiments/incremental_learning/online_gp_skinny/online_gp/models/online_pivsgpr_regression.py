import torch

from gpytorch.kernels import ScaleKernel, RBFKernel

from .streaming_sgpr import StreamingSGPR
from .online_sgpr_regression import OnlineSGPRegression
from ..utils.pivoted_cholesky import pivoted_cholesky_init

class PivCholStreamingSGPR(StreamingSGPR):
    def __init__(self, num_inducing_points=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if num_inducing_points is None:
            self.num_inducing_points = self.variational_strategy.inducing_points.shape[-2]
        else:
            self.num_inducing_points = num_inducing_points
        
    def _select_new_inducing_points(self, x_new, resample_ratio=0.2):
        old_inducing_points = self.variational_strategy.inducing_points.detach().clone()
        stacked_inducing_new = torch.cat((old_inducing_points, x_new), dim=-2)

        # concatenate new with old inducing points if we are less than our 
        # threshold
        if stacked_inducing_new.shape[-2] <= self.num_inducing_points:
            inducing_points = stacked_inducing_new
        else:
            inducing_points = pivoted_cholesky_init(
                stacked_inducing_new, self.covar_module(stacked_inducing_new), self.num_inducing_points
            )
            
        return inducing_points
        
    def get_fantasy_model(self, x_new, y_new, z_new=None, **kwargs):
        if z_new is None:
            z_new = self._select_new_inducing_points(x_new)
        res = super().get_fantasy_model(x_new, y_new, z_new, **kwargs)
        res.num_inducing_points = self.num_inducing_points
        return res
    
class OnlinePivSGPRegression(OnlineSGPRegression):
    def __init__(
            self,
            stem,
            init_x,
            init_y,
            num_inducing,
            lr,
            learn_inducing_locations=False,
            num_update_steps=1,
            covar_module=None,
            inducing_points=None,
            jitter=1e-4,
            **kwargs
        ):

        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel()).to(init_x.device)
            
        super().__init__(
            stem=stem,
            init_x=init_x,
            init_y=init_y,
            num_inducing=num_inducing,
            lr=lr,
            learn_inducing_locations=learn_inducing_locations,
            num_update_steps=num_update_steps,
            covar_module=covar_module,
            inducing_points=inducing_points,
            jitter=jitter,
            **kwargs,
        )
        self.stem = self.stem.to(init_x)
        if inducing_points is None:
            latent = self.stem(init_x).detach().clone()
            if num_inducing < init_x.shape[-2]:
                inducing_points = pivoted_cholesky_init(latent, covar_module(latent), num_inducing)
            else:
                inducing_points = latent
        
        # now reset the gp module
        self.gp = PivCholStreamingSGPR(
            inducing_points=inducing_points, 
            learn_inducing_locations=learn_inducing_locations,
            num_inducing_points=num_inducing,
            covar_module=covar_module, 
            num_data=init_x.size(-2), 
            jitter=jitter
        )