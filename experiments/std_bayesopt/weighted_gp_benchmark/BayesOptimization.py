# -*- coding: utf-8 -*-
"""
Contains the Bayes optimization class.
Initialization parameters:
    model: an object with methods 'predict', 'fit', and 'update'
    interface: an object which supplies the state of the system and
        allows for changing the system's x-value.
        Should have methods '(x,y) = intfc.getState()' and 'intfc.setX(x_new)'.
        Note that this interface system is rough, and used for testing and
            as a placeholder for the machine interface.
    acq_func: specifies how the optimizer should choose its next point.
        'EI': uses expected improvement. The interface should supply y-values.
        'testEI': uses EI over a finite set of points. This set must be 
            provided as alt_param, and the interface need not supply
            meaningful y-values.
    xi: exploration parameter suggested in some Bayesian opt. literature
    alt_param: currently only used when acq_func=='testEI'
    m: the maximum size of model; can be ignored unless passing an untrained
        SPGP or other model which doesn't already know its own size
    bounds: a tuple of (min,max) tuples specifying search bounds for each
        input dimension. Generally leads to better performance.
    prior_data: input data to train the model on initially. For convenience,
        since the model can be trained externally as well.
        Assumed to be a pandas DataFrame of shape (n, dim+1) where the last 
            column contains y-values.
            
Methods:
    acquire(): Returns the point that maximizes the acquisition function.
        For 'testEI', returns the index of the point instead.
        For normal acquisition, currently uses the bounded L-BFGS optimizer.
            Haven't tested alternatives much.
    best_seen(): Uses the model to make predictions at every observed point,
        returning the best-performing (x,y) pair. This is more robust to noise
        than returning the best observation, but could be replaced by other,
        faster methods.
    OptIter(): The main method for Bayesian optimization. Maximizes the
        acquisition function, then uses the interface to test this point and
        update the model.
"""

import operator as op
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class BayesOpt(object):
    def __init__(
        self,
        model,
        interface,
        acq_func="EI",
        xi=0.0,
        alt_param=-1,
        m=200,
        bounds=None,
        x_init=None,
        y_init=None,
        prior_data=None,
    ):
        self.model = model
        self.m = m
        self.bounds = bounds
        self.interface = interface
        self.acq_func = (acq_func, xi, alt_param)

        # if(acq_func=='testEI'):
        #    (x_init, y_init) = np.array(alt_param.iloc[0,:-1],ndmin=2),alt_param.iloc[0,-1]
        # else:
        #    (x_init, y_init) = interface.getState()

        self.X_obs = np.array(x_init)
        # print(max(y_init), "on init")
        self.Y_obs = [y_init]

        # initialize model on prior data
        if prior_data is not None:
            p_X = prior_data.iloc[:, :-1]
            p_Y = prior_data.iloc[:, -1]
            num = len(prior_data.index)

            self.model.fit(p_X, p_Y, min(m, num))

    def OptIter(self):
        # runs the optimizer for one iteration

        # get next point to try using acquisition function
        x_next = self.acquire()
        if self.acq_func[0] == "testEI":
            ind = x_next
            x_next = np.array(self.acq_func[2].iloc[ind, :-1], ndmin=2)

        # change position of interface and get resulting y-value
        self.interface.setX(x_next)
        if self.acq_func[0] == "testEI":
            (x_new, y_new) = (x_next, self.acq_func[2].iloc[ind, -1])
        else:
            (x_new, y_new) = self.interface.getState()

        # add new entry to observed data
        self.X_obs = np.concatenate((self.X_obs, x_new), axis=0)
        self.Y_obs.append(y_new)

        # update the model (may want to add noise if using testEI)
        self.model.update(x_new, y_new)  # + .5*np.random.randn())

    def best_seen(self):
        # checks the observed points to see which is predicted to be best.
        #   Probably safer than just returning the maximum observed, since the
        #       model has noise. It takes longer this way, though; you could
        #       instead take the model's prediction at the x-value that has
        #       done best if this needs to be faster.
        (mu, var) = self.model.predict(self.X_obs)

        (ind_best, mu_best) = max(enumerate(mu), key=op.itemgetter(1))
        return (self.X_obs[ind_best], mu_best)

    def acquire(self):
        # computes the next point for the optimizer to try
        if self.acq_func[0] == "EI":
            (x_best, y_best) = self.best_seen()
            print(y_best)
            # maximize the EI (by minimizing negative EI)
            try:
                res = minimize(
                    negExpImprove,
                    x_best,
                    args=(self.model, y_best, self.acq_func[1]),
                    bounds=self.bounds,
                    method="L-BFGS-B",
                    options={"maxfun": 100},
                )
            except:
                raise
            # return resulting x value as a (1 x dim) vector
            return np.array(res.x, ndmin=2)

        elif self.acq_func[0] == "testEI":
            # collect all possible x values
            options = np.array(self.acq_func[2].iloc[:, :-1])
            (x_best, y_best) = self.best_seen()

            # find the option with best EI
            best_option_score = (-1, 1e12)
            for i in range(options.shape[0]):
                result = negExpImprove(options[i], self.model, y_best, self.acq_func[1])
                if result < best_option_score[1]:
                    best_option_score = (i, result)

            # return the index of the best option
            return best_option_score[0]
        else:
            print("Unknown acquisition function.")
            return 0


def negExpImprove(x_new, model, y_best, xi):
    (y_new, var) = model.predict(np.array(x_new, ndmin=2))
    diff = y_new - y_best - xi
    if var == 0:
        return 0
    else:
        Z = diff / np.sqrt(var)

    EI = diff * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
    return -EI


def negProbImprove(x_new, model, y_best, xi):
    (y_new, var) = model.predict(np.array(x_new, ndmin=2))
    diff = y_new - y_best - xi
    if var == 0:
        return 0
    else:
        Z = diff / np.sqrt(var)

    return -norm.cdf(Z)
