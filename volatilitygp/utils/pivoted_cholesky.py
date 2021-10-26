import torch

from torch import Tensor
from gpytorch.lazy import LazyTensor

from typing import Union


def pivoted_cholesky_init(
    train_inputs: Tensor,
    kernel_matrix: Union[Tensor, LazyTensor],
    max_length: int,
    epsilon: float = 1e-10,
) -> Tensor:
    r"""
    A pivoted cholesky initialization method for the inducing points, originally proposed in
    [burt2020svgp] with the algorithm itself coming from [chen2018dpp]. Code is a PyTorch version from
    [chen2018dpp], copied from https://github.com/laming-chen/fast-map-dpp/blob/master/dpp.py.
    Args:
        train_inputs [Tensor]: training inputs
        kernel_matrix [Tensor or Lazy Tensor]: kernel matrix on the training inputs
        max_length [int]: number of inducing points to initialize
        epsilon [float]: numerical jitter for stability.
    """
    # this is numerically equivalent to iteratively performing a pivoted cholesky
    # while storing the diagonal pivots at each iteration
    # TODO: use gpytorch's pivoted cholesky instead once that gets an exposed list
    # TODO: setup a non for looped batch mode
    if kernel_matrix.ndim > 2:
        # if we have batch dimensions, recurse
        piv_chol_list = []
        for i in range(kernel_matrix.shape[0]):
            piv_chol_list.append(
                pivoted_cholesky_init(
                    train_inputs[i], kernel_matrix[i], max_length, epsilon
                ).unsqueeze(0)
            )
        res = torch.cat(piv_chol_list, dim=0)
        return res

    item_size = kernel_matrix.shape[-2]
    cis = torch.zeros((max_length, item_size)).to(kernel_matrix.device)
    di2s = kernel_matrix.diag()
    selected_items = []
    selected_item = torch.argmax(di2s)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = torch.sqrt(di2s[selected_item])
        elements = kernel_matrix[..., selected_item, :]
        eis = (elements - torch.matmul(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s = di2s - eis.pow(2.0)
        di2s[selected_item] = -(torch.tensor(float("inf")))
        selected_item = torch.argmax(di2s)
        if di2s[selected_item] < epsilon:
            break
        selected_items.append(selected_item)
    ind_points = train_inputs[torch.stack(selected_items)]
    return ind_points
