import torch

from torch import Tensor
from gpytorch.lazy import LazyTensor
from gpytorch.kernels import ScaleKernel, RBFKernel
from typing import Union

from .variational_gp_model import VariationalGPModel
from ..utils.pivoted_cholesky import pivoted_cholesky_init

class PivCholVariationalGPModel(VariationalGPModel):
    def __init__(
        self,
        train_x,
        num_inducing_points=None,
        mean_module=None,
        covar_module=None,
        streaming=False,
        likelihood=None,
        feat_extractor=None,
        beta=1.0,
        learn_inducing_locations=False,
    ):
        if num_inducing_points is None:
            num_inducing_points = int(train_x.shape[0] / 2)

        if covar_module is None:
            covar_module = ScaleKernel(RBFKernel())

        if num_inducing_points < train_x.shape[-2]:
            inducing_points = pivoted_cholesky_init(train_x, covar_module(train_x), num_inducing_points)
        else:
            inducing_points = train_x.detach().clone()

        super().__init__(
            inducing_points=inducing_points,
            mean_module=mean_module,
            covar_module=covar_module,
            streaming=streaming,
            likelihood=likelihood,
            feat_extractor=feat_extractor,
            beta=beta,
            learn_inducing_locations=learn_inducing_locations,
        )

        self.num_inducing_points = num_inducing_points

    def update_variational_parameters(self, new_x, new_y, new_inducing_points=None):
        if new_inducing_points is None:
            # here, we stack the inducing points and the new ones while re-computing the 
            # pivoted cholesky with these points as an approximation to computing the 
            # pivoted cholesky of the [training data, new data]
            old_inducing_points = self.variational_strategy.inducing_points.detach().clone()
            stacked_inducing_new = torch.cat((old_inducing_points, new_x), dim=-2)

            # concatenate new with old inducing points if we are less than our 
            # threshold
            if stacked_inducing_new.shape[-2] <= self.num_inducing_points:
                inducing_points = stacked_inducing_new
            else:
                inducing_points = pivoted_cholesky_init(
                    stacked_inducing_new, self.covar_module(stacked_inducing_new), self.num_inducing_points
                )

        return super().update_variational_parameters(new_x, new_y, inducing_points)
