# -*- coding: utf-8 -*-
"""
Designed by Lehel Csato for NETLAB, rewritten
for Python in 2016 by Mitchell McIntire

The Online Gaussian process class.

Initialization parameters:
    dim: the dimension of input data
    hyperparams: GP model hyperparameters. For RBF_ARD, a 3-tuple with entries:
        hyp_ARD: size (1 x dim) vector of ARD parameters
        hyp_coeff: the coefficient parameter of the RBF kernel
        hyp_noise: the model noise hyperparameter
        Note -- different hyperparams needed for different covariance functions
    covar: the covariance function to be used, currently only 'RBF_ARD'
        RBF_ARD: the radial basis function with ARD, i.e. a squared exponential
            with diagonal scaling matrix specified by hyp_ARD
    maxBV: the number of basis vectors to represent the model. Increasing this
        beyond 100 or so will increase runtime substantially
    prmean: either None, a number, or a callable function that gives the prior mean
    prmeanp: parameters to the prmean function
    proj: I'm not sure exactly. Setting this to false gives a different method
        of computing updates, but I haven't tested it or figured out what the
        difference is.
    weighted: whether or not to use weighted difference computations. Slower but may
        yield improved performance. Still testing.
    thresh: some low float value to specify how different a point has to be to
        add it to the model. Keeps matrices well-conditioned.
        
Methods:
    update(x_new, y_new): Runs an online GP iteration incorporating the new data.
    fit(X, Y): Calls update on multiple points for convenience. X is assumed to
        be a pandas DataFrame.
    predict(x): Computes GP prediction(s) for input point(s).
    scoreBVs(): Returns a vector with the (either weighted or unweighted) KL
        divergence-cost of removing each BV.
    deleteBV(index): Removes the selected BV from the GP and updates to minimize
        the (either weighted or unweighted) KL divergence-cost of the removal
"""
import torch
import math
from gpytorch.kernels import RBFKernel, ScaleKernel

# import numpy as np
import numbers

# from numpy.linalg import solve, inv


class OGP(object):
    def __init__(
        self,
        dim,
        hyperparams,
        covar="RBF_ARD",
        maxBV=200,
        prmean=None,
        prmeanp=None,
        proj=True,
        weighted=False,
        thresh=1e-6,
    ):
        self.nin = dim
        self.maxBV = maxBV
        self.numBV = 0
        self.proj = proj
        self.weighted = weighted

        #         if(covar in ['RBF_ARD']):
        #             self.covar = covar
        #             self.covar_params = hyperparams[:2]
        #         else:
        #             print('Unknown covariance function')
        #             raise
        self.kernel = ScaleKernel(RBFKernel(ard_num_dims=dim)).double()
        self.kernel.base_kernel.lengthscale = torch.tensor(hyperparams[0]).exp()
        self.kernel.base_kernel.outputscale = torch.tensor(hyperparams[1]).exp()

        self.noise_var = torch.exp(torch.tensor(hyperparams[2]))

        self.prmean = prmean
        self.prmeanp = prmeanp

        # initialize model state
        self.BV = torch.zeros(size=(0, self.nin)).double()
        self.alpha = torch.zeros(size=(0, 1)).double()
        self.C = torch.zeros(size=(0, 0)).double()

        self.KB = torch.zeros(size=(0, 0)).double()
        self.KBinv = torch.zeros(size=(0, 0)).double()

        self.thresh = thresh

    def fit(self, X, Y, m=0):
        # just train on all the data in X. m is a dummy parameter
        for i in range(X.shape[0]):
            self.update(torch.tensor(X.iloc[i, :]).view(1, -1), Y[i])

    def update(self, x_new, y_new):
        # compute covariance with BVs
        k_x = self.kernel(self.BV, x_new).evaluate()
        k = self.kernel(x_new).evaluate()
        # k_x = self.computeCov(self.BV, x_new)
        # k = self.computeCov(x_new, x_new)

        # compute mean and variance
        cM = torch.matmul(k_x.t(), self.alpha)
        cV = k + torch.matmul(k_x.t(), torch.matmul(self.C, k_x))
        # not needed if nout==1: cV = (cV + np.transpose(cV)) / 2
        # cV = torch.max((cV, 1e-12))
        cV = cV.clamp(min=1e-12)

        pM = self.priorMean(x_new)

        (logLik, K1, K2) = logLikelihood(self.noise_var, y_new, cM + pM, cV)

        # compute gamma, a geometric measure of novelty
        if self.KB.shape[0] > 0:
            hatE = torch.solve(k_x, self.KB)[0]
            gamma = k - torch.matmul(k_x.t(), hatE)
        else:
            hatE = torch.zeros(0, 1)
            gamma = torch.ones(1, 1)

        if gamma < self.thresh * k:
            # not very novel, just tweak parameters
            self._sparseParamUpdate(k_x, K1, K2, gamma, hatE)
        else:
            # expand model
            self._fullParamUpdate(x_new, k_x, k, K1, K2, gamma, hatE)

        # reduce model according to maxBV constraint
        while self.BV.shape[0] > self.maxBV:
            minBVind = self.scoreBVs()
            self.deleteBV(minBVind)

    def predict(self, x_in):
        if not isinstance(x_in, torch.Tensor):
            x_in = torch.tensor(x_in)
        # reads in a (n x dim) vector and returns the (n x 1) vector
        #   of predictions along with predictive variance for each

        # k_x = self.computeCov(x_in, self.BV)
        # k = self.computeCov(x_in, x_in)
        k_x = self.kernel(x_in, self.BV).evaluate()
        k = self.kernel(x_in).evaluate()
        pred = torch.matmul(k_x, self.alpha)
        var = k + torch.matmul(k_x, torch.matmul(self.C, k_x.t()))

        pmean = self.priorMean(x_in)
        return pmean + pred, var

    def _sparseParamUpdate(self, k_x, K1, K2, gamma, hatE):
        # computes a sparse update to the model without expanding parameters

        eta = 1
        if self.proj:
            eta += K2 * gamma

        CplusQk = torch.matmul(self.C, k_x) + hatE
        self.alpha = self.alpha + (K1 / eta) * CplusQk
        eta = K2 / eta
        self.C = self.C + eta * torch.matmul(CplusQk, CplusQk.t())
        self.C = stabilizeMatrix(self.C)

    def _fullParamUpdate(self, x_new, k_x, k, K1, K2, gamma, hatE):
        # print(self.KBinv.shape, gamma.shape, hatE.shape)
        # expands parameters to incorporate new input

        # add new input to basis vectors
        oldnumBV = self.BV.shape[0]
        numBV = oldnumBV + 1
        self.BV = torch.cat((self.BV, x_new), 0)

        hatE = extendVector(hatE, val=-torch.ones(1, 1))

        # update KBinv
        self.KBinv = extendMatrix(self.KBinv)
        # print(self.KBinv.shape, "after")
        self.KBinv = self.KBinv + (1 / gamma) * torch.matmul(hatE, hatE.t())
        # print(self.KBinv.shape, "after this op")

        # update Gram matrix
        self.KB = extendMatrix(self.KB)
        if numBV > 1:
            self.KB[0:oldnumBV, [oldnumBV]] = k_x
            self.KB[[oldnumBV], 0:oldnumBV] = k_x.t()
        self.KB[oldnumBV, oldnumBV] = k

        Ck = extendVector(torch.matmul(self.C, k_x), val=torch.ones(1, 1))

        self.alpha = extendVector(self.alpha)
        self.C = extendMatrix(self.C)

        self.alpha = self.alpha + K1 * Ck
        self.C = self.C + K2 * torch.matmul(Ck, Ck.t())

        # stabilize matrices for conditioning/reducing floating point errors?
        self.C = stabilizeMatrix(self.C)
        self.KB = stabilizeMatrix(self.KB)
        self.KBinv = stabilizeMatrix(self.KBinv)
        # print(self.KB.shape, self.KBinv.shape, "at end")

    def scoreBVs(self):
        # measures the importance of each BV for model accuracy
        # currently quite slow for the weighted GP if numBV is much more than 50

        numBV = self.BV.shape[0]
        a = self.alpha
        if not self.weighted:
            scores = (a * a).reshape((numBV)) / (
                self.C.diagonal() + self.KBinv.diagonal()
            )
        else:
            scores = np.zeros(shape=(numBV, 1))

            # This is slow, in particular the numBV calls to computeWeightedDiv
            for removed in range(numBV):
                (hatalpha, hatC) = self.getUpdatedParams(removed)

                scores[removed] = self.computeWeightedDiv(hatalpha, hatC, removed)

        return scores.argmin()

    def priorMean(self, x):
        if callable(self.prmean):
            if self.prmeanp is not None:
                return self.prmean(x, self.prmeanp)
            else:
                return self.prmean(x)
        elif isinstance(self.prmean, numbers.Number):
            return self.prmean
        else:
            # if no prior mean function is supplied, assume zero
            return 0

    def deleteBV(self, removeInd):
        # removes a BV from the model and modifies parameters to
        #   attempt to minimize the removal's impact

        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]

        # update alpha and C
        (self.alpha, self.C) = self.getUpdatedParams(removeInd)

        # stabilize C
        self.C = stabilizeMatrix(self.C)

        # update KB and KBinv
        q_star = self.KBinv[removeInd, removeInd]
        red_q = self.KBinv[keepInd][:, [removeInd]]
        self.KBinv = self.KBinv[keepInd][:, keepInd] - (1 / q_star) * torch.matmul(
            red_q, red_q.t()
        )
        self.KBinv = stabilizeMatrix(self.KBinv)

        self.KB = self.KB[keepInd][:, keepInd]
        self.BV = self.BV[keepInd]

    def computeWeightedDiv(self, hatalpha, hatC, removeInd):
        # computes the weighted divergence for removing a specific BV
        # currently uses matrix inversion and therefore somewhat slow

        hatalpha = extendVector(hatalpha, ind=removeInd)
        hatC = extendMatrix(hatC, ind=removeInd)

        diff = self.alpha - hatalpha
        scale = np.dot(self.alpha.transpose(), np.dot(self.KB, self.alpha))

        Gamma = np.eye(self.BV.shape[0]) + np.dot(self.KB, self.C)
        Gamma = Gamma.transpose() / scale + torch.eye(self.BV.shape[0])
        M = 2 * torch.matmul(Gamma, self.alpha) - (self.alpha + hatalpha)

        hatV = torch.inverse(hatC + self.KBinv)
        (s, logdet) = torch.logdet(torch.matmul(self.C + self.KBinv, hatV))

        if s == 1:
            w = torch.sum(torch.matmul(self.C - hatC, hatV).diag()) - logdet
            # w = np.trace(np.dot(self.C - hatC, hatV)) - logdet
        else:
            w = np.Inf

        return torch.matmul(M.t(), torch.matmul(hatV, diff)) + w

    def getUpdatedParams(self, removeInd):
        # computes updates for alpha and C after removing the given BV

        numBV = self.BV.shape[0]
        keepInd = [i for i in range(numBV) if i != removeInd]
        a = self.alpha

        if not self.weighted:
            # compute auxiliary variables
            q_star = self.KBinv[removeInd, removeInd]
            red_q = self.KBinv[keepInd][:, [removeInd]]
            c_star = self.C[removeInd, removeInd]
            red_CQsum = red_q + self.C[keepInd][:, [removeInd]]

            if self.proj:
                hatalpha = a[keepInd] - (a[removeInd] / (q_star + c_star)) * red_CQsum
                hatC = (
                    self.C[keepInd][:, keepInd]
                    + (1 / q_star) * torch.matmul(red_q, red_q.t())
                    - (1 / (q_star + c_star)) * torch.matmul(red_CQsum, red_CQsum.t())
                )
            else:
                tempQ = red_q / q_star
                hatalpha = a[keepInd] - a[removeInd] * tempQ
                red_c = self.C[removeInd, [keepInd]]
                hatC = self.C[keepInd][:, keepInd] + c_star * torch.matmul(
                    tempQ, tempQ.t()
                )
                tempQ = torch.matmul(tempQ, red_c)
                hatC = hatC - tempQ - tempQ.t()
        else:
            # compute auxiliary variables
            q_star = self.KBinv[removeInd, removeInd]
            red_q = self.KBinv[keepInd][:, [removeInd]]
            c_star = self.C[removeInd, removeInd]
            red_CQsum = red_q + self.C[keepInd][:, [removeInd]]
            Gamma = (torch.eye(numBV) + torch.matmul(self.KB, self.C)).t()
            Gamma = torch.eye(numBV) + Gamma / torch.matmul(
                a.t(), torch.matmul(self.KB, a)
            )

            hatalpha = (
                torch.matmul(Gamma[keepInd], a)
                - torch.matmul(Gamma[removeInd], a) * red_q / q_star
            )

            # this isn't rigorous...
            # extend = extendVector(hatalpha, ind=removeInd)
            hatC = self.C  # + np.dot(2*np.dot(Gamma,a) - (a + extend),
            # (a - extend).transpose())
            hatC = (
                hatC[keepInd][:, keepInd]
                + (1 / q_star) * torch.matmul(red_q, red_q.t())
                - (1 / (q_star + c_star)) * torch.matmul(red_CQsum, red_CQsum.t())
            )

        return hatalpha, hatC

    def computeCov(self, x1, x2, is_self=False):
        # computes covariance between inputs x1 and x2
        #   returns a matrix of size (n1 x n2)

        (n1, dim) = x1.shape
        n2 = x2.shape[0]

        (hyp_ARD, hyp_coeff) = self.covar_params

        b = np.exp(hyp_ARD)
        coeff = np.exp(hyp_coeff)

        # use ARD to scale
        b_sqrt = np.sqrt(b)
        x1 = x1 * b_sqrt
        x2 = x2 * b_sqrt

        x1_sum_sq = np.reshape(np.sum(x1 * x1, axis=1), (n1, 1))
        x2_sum_sq = np.reshape(np.sum(x2 * x2, axis=1), (1, n2))

        K = -2 * np.dot(x1, x2.transpose())
        K = K + x1_sum_sq + x2_sum_sq
        K = coeff * np.exp(-0.5 * K)

        if is_self:
            jitter = 1e-6
            K = K + jitter * np.eye(n1)

        return K


def logLikelihood(noise, y, mu, var):
    sigX2 = noise + var
    K2 = -1 / sigX2
    K1 = -K2 * (y - mu)
    logLik = -(torch.log(2 * math.pi * sigX2) + (y - mu) * K1) / 2

    return logLik, K1, K2


def stabilizeMatrix(M):
    return (M + M.t()) / 2


def extendMatrix(M, ind=-1):
    if ind == -1:
        M = torch.cat((M, torch.zeros(size=(M.shape[0], 1))), dim=1)
        M = torch.cat((M, torch.zeros(size=(1, M.shape[1]))), dim=0)
    elif ind == 0:
        M = torch.cat((torch.zeros(size=(M.shape[0], 1)), M), dim=1)
        M = torch.cat((torch.zeros(size=(1, M.shape[1])), M), dim=0)
    else:
        M = torch.cat((M[:ind], torch.zeros(size=(1, M.shape[1])), M[ind:]), dim=0)
        M = torch.cat((M[:, :ind], torch.zeros(size=(M.shape[0], 1)), M[:, ind:]), dim=1)
    return M


def extendVector(v, val=torch.zeros(1, 1), ind=-1):
    if ind == -1:
        return torch.cat((v, val), dim=0)
    elif ind == 0:
        return torch.cat((val, v), dim=0)
    else:
        return torch.cat((v[:ind], val, v[ind:]), dim=0)
