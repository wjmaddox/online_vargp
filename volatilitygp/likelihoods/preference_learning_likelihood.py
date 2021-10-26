import torch

from volatilitygp.likelihoods._one_dimensional_likelihood import (
    NewtonOneDimensionalLikelihood,
)
from gpytorch.constraints import Interval
from gpytorch import lazify
from torch.distributions import Distribution, Normal

base_normal_logcdf = (
    lambda x: Normal(torch.zeros(1).to(x), torch.ones(1).to(x)).cdf(x).log()
)
base_normal_logpdf = lambda x: Normal(torch.zeros(1).to(x), torch.ones(1).to(x)).log_prob(
    x
)


def _prepare_scatter_matrices(n, dtype, targets):
    # prepare S_k(x_i) from Chu
    # https://github.com/pytorch/botorch/blob/913aa0e510dde10568c2b4b911124cdd626f6905/botorch/models/pairwise_gp.py#L761
    if targets.ndim == 1:
        targets = targets.unsqueeze(0)
    m = targets.shape[-2]
    sign_term = torch.zeros(*targets.shape[:-1], n, dtype=dtype, device=targets.device)
    comp_view = targets.view(-1, m, 2).long()
    for i, sub_D in enumerate(sign_term.view(-1, m, n)):
        sub_D.scatter_(1, comp_view[i, :, [0]], 1)
        sub_D.scatter_(1, comp_view[i, :, [1]], -1)
    # sign_term_t = sign_term.transpose(-1, -2)
    return sign_term  # , sign_term_t


class PrefLearningDist(Distribution):
    def __init__(self, f_vals, noise):
        super().__init__()

        self.f_vals = f_vals
        self.noise = noise

    def _prepare_z_k_and_scatter(self, f_vals, comparisons, noise):
        sign_term = _prepare_scatter_matrices(f_vals.shape[-1], f_vals.dtype, comparisons)

        # from https://github.com/pytorch/botorch/blob/913aa0e510dde10568c2b4b911124cdd626f6905/botorch/models/pairwise_gp.py#L315

        scaled_utility = (f_vals / (noise * 2) ** 0.5).unsqueeze(-1)
        utility = sign_term.matmul(scaled_utility).squeeze(-1)
        utility = utility.clamp(min=-3.0, max=3.0)
        return utility, sign_term

    def log_prob(self, comparisons, sum_over_prefs=False):
        z_k, sign_term = self._prepare_z_k_and_scatter(
            self.f_vals, comparisons, self.noise
        )
        if sum_over_prefs:
            return (
                -sign_term.transpose(-1, -2)
                .matmul(base_normal_logcdf(z_k).unsqueeze(-1))
                .sum(-1)
            )
        else:
            return base_normal_logcdf(z_k)

    def sample(self, shape=torch.Size()):
        n_comp = 2 * self.f_vals.shape[-1]
        s1 = torch.randint(0, self.f_vals.shape[-1], size=torch.Size((n_comp,)))
        s2 = torch.randint(0, self.f_vals.shape[-1], size=torch.Size((n_comp,)))
        s1[s1 == s2] = (s1[s1 == s2] - 1).clamp(min=0)
        s1_f = self.f_vals[..., s1] + torch.randn_like(self.f_vals[..., s1]) * (
            self.noise ** 0.5
        )
        s2_f = self.f_vals[..., s2] + torch.randn_like(self.f_vals[..., s2]) * (
            self.noise ** 0.5
        )

        comp_pairs = torch.cat((s1.unsqueeze(-1), s2.unsqueeze(-1)), dim=-1)
        comp_pairs = comp_pairs.expand(*self.f_vals.shape[:-1], *comp_pairs.shape)
        lt_comps = s1_f < s2_f
        old_comp = comp_pairs[..., 0].clone()[lt_comps]
        comp_pairs[..., 0][lt_comps] = comp_pairs[..., 1].clone()[lt_comps]
        comp_pairs[..., 1][lt_comps] = old_comp
        return comp_pairs

    # TODO: rsample


class PrefLearningLikelihood(NewtonOneDimensionalLikelihood):
    has_diag_hessian = False

    def __init__(self, batch_shape=torch.Size(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_noise = torch.nn.Parameter(torch.zeros(*batch_shape, 1))

        self.register_constraint("raw_noise", Interval(1e-3, 1.0))

    @property
    def noise(self):
        return self.raw_noise_constraint.transform(self.raw_noise)

    def forward(self, function_samples, *args, **kwargs):
        return PrefLearningDist(function_samples, noise=self.noise)

    def grad_f(self, f, targets):
        z_k, sign_term = PrefLearningDist._prepare_z_k_and_scatter(None, f, targets, 1.0)
        grad_term = (base_normal_logpdf(z_k) - base_normal_logcdf(z_k)).exp() / (2) ** 0.5
        return grad_term.unsqueeze(-2).matmul(sign_term).squeeze(-2)

    def neg_hessian_f(self, f, targets=None):
        if targets is not None:
            if targets.shape[-1] != f.shape[-1] or targets.shape[-1] != 1:
                targets = None
        if targets is None:
            # emulate targets via posterior
            targets_dist = self.forward(f)
            targets = targets_dist.sample()

        z_k, sign_term = PrefLearningDist._prepare_z_k_and_scatter(
            None, f, targets, self.noise
        )
        sign_term_t = sign_term.transpose(-1, -2)
        log_pdf_cdf_ratio = base_normal_logpdf(z_k) - base_normal_logcdf(z_k)

        term1 = log_pdf_cdf_ratio.exp() + z_k
        grad_term = log_pdf_cdf_ratio.exp() / (2 * self.noise)

        inner_term = term1 * grad_term
        expanded_inner_term = inner_term.unsqueeze(-2).expand(sign_term_t.shape)
        weighted_sign_term = sign_term_t * expanded_inner_term
        res = weighted_sign_term.matmul(sign_term)
        res = (res + res.transpose(-1, -2)) / 2
        return lazify(res).add_jitter(1e-5).evaluate()
