from gpytorch.lazy import DiagLazyTensor
from gpytorch.likelihoods import (
    FixedNoiseGaussianLikelihood as _FixedNoiseGaussianLikelihood,
)


class FixedNoiseGaussianLikelihood(_FixedNoiseGaussianLikelihood):
    def newton_iteration(self, *args, **kwargs):
        pass

    def expected_hessian(self, *args, **kwargs):
        return DiagLazyTensor(self.noise).inverse()
