from itertools import combinations

import argparse
import numpy as np
import torch
import math

from scipy.stats import kendalltau
from botorch.models import PairwiseGP, PairwiseLaplaceMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model
from botorch.optim.fit import fit_gpytorch_torch
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors import SmoothedBoxPrior, GammaPrior

from volatilitygp.models import SingleTaskVariationalGP
from volatilitygp.likelihoods import PrefLearningLikelihood

# taken from https://github.com/pytorch/botorch/blob/master/tutorials/preference_bo.ipynb


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="results.pt")
    # parser.add_argument("--problem", type=str, default="hartmann6")
    parser.add_argument("--n_batch", type=int, default=50)
    # parser.add_argument("--mc_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--method", type=str, default="laplace")
    return parser.parse_args()


# data generating helper functions
def utility(X):
    """ Given X, output corresponding utility (i.e., the latent function)
    """
    # y is weighted sum of X, with weight sqrt(i) imposed on dimension i
    weighted_X = (
        -(X * 2 * math.pi).cos()
        * torch.sqrt(torch.arange(X.size(-1), dtype=torch.float) + 1)
        / (X.shape[-1] ** 0.5)
    )
    y = torch.sum(weighted_X, dim=-1)
    return y


def generate_data(n, dim=2):
    """ Generate data X and y """
    # X is randomly sampled from dim-dimentional unit cube
    # we recommend using double as opposed to float tensor here for
    # better numerical stability
    X = torch.rand(n, dim, dtype=torch.float64)
    y = utility(X)
    return X, y


def generate_comparisons(y, n_comp, noise=0.1, replace=False):
    """  Create pairwise comparisons with noise """
    # generate all possible pairs of elements in y
    all_pairs = np.array(list(combinations(range(y.shape[0]), 2)))
    # randomly select n_comp pairs from all_pairs
    comp_pairs = all_pairs[
        np.random.choice(range(len(all_pairs)), n_comp, replace=replace)
    ]
    # add gaussian noise to the latent y values
    c0 = y[comp_pairs[:, 0]] + np.random.standard_normal(len(comp_pairs)) * noise
    c1 = y[comp_pairs[:, 1]] + np.random.standard_normal(len(comp_pairs)) * noise
    reverse_comp = (c0 < c1).numpy()
    comp_pairs[reverse_comp, :] = np.flip(comp_pairs[reverse_comp, :], 1)
    comp_pairs = torch.tensor(comp_pairs).long()

    return comp_pairs


# Kendall-Tau rank correlation
def eval_kt_cor(model, test_X, test_y):
    pred_y = model.posterior(test_X).mean.squeeze().detach().numpy()
    return kendalltau(pred_y, test_y).correlation


def make_new_data(X, next_X, comps, q_comp, noise=0.1):
    """ Given X and next_X, 
    generate q_comp new comparisons between next_X
    and return the concatenated X and comparisons
    """
    # next_X is float by default; cast it to the dtype of X (i.e., double)
    next_X = next_X.to(X)
    next_y = utility(next_X)
    next_comps = generate_comparisons(next_y, n_comp=q_comp, noise=noise)
    comps = torch.cat([comps, next_comps + X.shape[-2]])
    X = torch.cat([X, next_X])
    return X, comps


def init_and_fit_model(X, comp, method="laplace"):
    if method == "laplace":
        model = PairwiseGP(X, comp)
        mll = PairwiseLaplaceMarginalLogLikelihood(model)
        fit_gpytorch_model(mll)
    elif method == "variational":
        print(X.dtype, comp.dtype)
        model = SingleTaskVariationalGP(
            likelihood=PrefLearningLikelihood(),
            init_points=X,
            use_piv_chol_init=True,
            init_targets=comp,
            num_inducing=25,
            covar_module=ScaleKernel(
                MaternKernel(
                    ard_num_dims=X.shape[-1], lengthscale_prior=GammaPrior(1.2, 0.5)
                ),
                outputscale_prior=SmoothedBoxPrior(a=1, b=4),
            ),
        )
        mll = VariationalELBO(model.likelihood, model, num_data=comp.shape[-2])
        fit_gpytorch_torch(mll, options={"maxiter": 1000})
    return mll, model
