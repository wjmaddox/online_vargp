import math
from dataclasses import dataclass

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from botorch.sampling import IIDNormalSampler
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from torch.quasirandom import SobolEngine

from numpy import unique as np_unique

# this class comes from https://botorch.org/tutorials/turbo_1


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_candidates(x_center, dim, n_candidates, bounds, dtype, device):
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = bounds[0] + (bounds[1] - bounds[0]) * pert

    # Create a perturbation mask
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1

    # Create candidate points from the perturbations and the mask
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    return X_cand


def generate_batch(
    state,
    model,  # GP model
    X,  # Evaluated points on the domain [0, 1]^d
    Y,  # Function values
    batch_size,
    n_candidates=None,  # Number of candidates for Thompson sampling
    num_restarts=10,
    raw_samples=512,
    tree_depth=4,
    acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei", "gibbon", "ts2", "ts_rollout")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    dtype = X.dtype
    device = X.device

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    dim = X.shape[-1]

    if acqf == "ts":

        X_cand = generate_candidates(
            x_center,
            dim,
            n_candidates,
            torch.stack([tr_lb, tr_ub]),
            device=device,
            dtype=dtype,
        )

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, train_Y.max(), maximize=True)
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
    elif acqf == "gibbon":
        # to force fantasization, we devote half the batch to Thompson sampling
        # and half the batch to gibbon
        ts_queried_points = generate_batch(
            state,
            model,  # GP model
            X,  # Evaluated points on the domain [0, 1]^d
            Y,  # Function values
            int(batch_size / 2),
            n_candidates=n_candidates,  # Number of candidates for Thompson sampling
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            acqf="ts",
        )
        bounds = torch.stack([tr_lb, tr_ub])
        candidate_set = torch.rand(
            n_candidates, bounds.shape[-1], device=device, dtype=dtype
        )
        candidate_set = bounds[0] + (bounds[1] - bounds[0]) * candidate_set
        gibbon = qLowerBoundMaxValueEntropy(
            model, candidate_set=candidate_set, X_pending=ts_queried_points
        )
        X_next, acq_value = optimize_acqf(
            gibbon,
            bounds=bounds,
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            sequential=True,
        )
        X_next = torch.cat((X_next, ts_queried_points.detach()))
    elif acqf == "ts2":
        X_next = generate_batch(
            state, model, X, Y, int(batch_size / 2), n_candidates=n_candidates, acqf="ts",
        )
        sampler = IIDNormalSampler(num_samples=1)
        fantasy_model = model.fantasize(X_next, sampler=sampler)
        ### NOTE TO SELF: you don't want to update the trust regions until you've actually used them
        fantasy_X = generate_batch(
            state,
            fantasy_model,
            X,
            Y,
            int(batch_size / 2),
            n_candidates=n_candidates,
            acqf="ts",
        )
        if fantasy_X.ndim == 3:
            fantasy_X = fantasy_X[0]

        X_next = torch.cat((X_next, fantasy_X))

    elif acqf == "ts_rollout":
        # tree_depth = 4
        orig_topk = 10
        rollout_candidates = int(n_candidates / tree_depth)
        candidates = [
            generate_candidates(
                x_center,
                dim,
                rollout_candidates,
                torch.stack([tr_lb, tr_ub]),
                device=device,
                dtype=dtype,
            )
        ] * tree_depth
        with torch.no_grad():
            inputs, targets = rollout_path(candidates, model, tree_depth, orig_topk)
            inputs = inputs.reshape(-1, inputs.shape[-1])
            new_inputs = inputs.unique(dim=0)
            inds = torch.tensor(
                np_unique(inputs.detach().cpu(), axis=0, return_index=True)[1]
            )
            targets = targets.reshape(-1).clone()
            new_targets = targets[inds].contiguous()

            fantasy_model = model.condition_on_observations(new_inputs, new_targets)
        ### NOTE TO SELF: you don't want to update the trust regions until you've actually used them
        # all we want to do is to update the model
        X_next = generate_batch(
            state, fantasy_model, X, Y, batch_size, n_candidates=n_candidates, acqf="ts"
        )

        if X_next.ndim == 3:
            X_next = X_next[0]

    return X_next


def rollout_path(candidate_sets, model, tree_depth, orig_topk):
    post_samples = model.posterior(candidate_sets[0]).rsample().squeeze(0)
    samples, inds = torch.topk(post_samples, dim=-2, k=orig_topk)

    if candidate_sets[0].ndim < 3:
        cs = candidate_sets[0][inds]
    else:
        cs = torch.stack([candidate_sets[0][i, inds[i]] for i in range(inds.shape[0])])

    fantasy = model.condition_on_observations(cs, samples.unsqueeze(-1))
    for depth in range(tree_depth - 1):
        post_samples = fantasy.posterior(candidate_sets[depth + 1]).rsample().squeeze(0)
        samples, inds = torch.topk(post_samples, dim=-2, k=1)
        fantasy = fantasy.condition_on_observations(
            candidate_sets[depth + 1][inds].squeeze(-2), samples,
        )
    targets = fantasy.train_targets
    return fantasy.train_inputs[0][..., -tree_depth:, :], targets[..., -tree_depth:]
