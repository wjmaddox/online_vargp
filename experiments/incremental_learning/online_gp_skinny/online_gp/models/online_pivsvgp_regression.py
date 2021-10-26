import torch

from .online_svgp_regression import OnlineSVGPRegression
from .chol_variational_gp_model import PivCholVariationalGPModel

class OnlinePivSVGPRegression(OnlineSVGPRegression):
    def __init__(
        self,
        stem,
        init_x,
        init_y,
        num_inducing,
        lr,
        streaming=False,
        prior_beta=1.,
        online_beta=1.,
        learn_inducing_locations=False,
        num_update_steps=1,
        covar_module=None,
        inducing_points=None,
        **kwargs
    ):
        super().__init__(
            stem=stem, 
            init_x=init_x, 
            init_y=init_y, 
            num_inducing=num_inducing, 
            lr=lr, 
            streaming=streaming,
            prior_beta=prior_beta,
            online_beta=online_beta,
            learn_inducing_locations=learn_inducing_locations,
            num_update_steps=num_update_steps,
            covar_module=covar_module,
            inducing_points=None,
            **kwargs
        )

        # now we reset the gp module
        likelihood = self.gp.likelihood
        mean_module = self.gp.mean_module
        self.gp = PivCholVariationalGPModel(
            train_x=self.stem(init_x.cpu()).detach(),
            num_inducing_points=num_inducing, 
            mean_module=mean_module, 
            covar_module=covar_module, 
            streaming=streaming, 
            likelihood=likelihood,
            beta=online_beta, 
            learn_inducing_locations=learn_inducing_locations
        ) 