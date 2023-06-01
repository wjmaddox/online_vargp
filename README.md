# Conditioning Sparse Variational Gaussian Processes for Online Decision-making

This repository contains a PyTorch and GPyTorch implementation of the paper ["Conditioning Sparse Variational Gaussian Processes for Online Decision-making"](https://openreview.net/forum?id=CCvpHGFOzC3) (NeurIPS 2021).

## Introduction

Online variational conditioning (OVC) provides closed form conditioning (e.g. updating a model's posterior predictive distribution after having observed new data points) for stochastic variational Gaussian processes. 
OVC enables the development of ``fantasization" (predicting on data and then conditioning on a random posterior sample)
for variational GPs, thereby enabling SVGPs to be used for the first time in advanced, look-ahead acquisitions 
such as the batch knowledge gradient, entropy search, and look-ahead Thompson sampling (which we introduce).

In this repo, we provide an implementation of a SVGP model with OVC hooked up as the `get_fantasy_model` function, allowing
it to be natively used with any advanced acquisition function in BoTorch (see the experiments in the `experiments/std_bayesopt` folder).

## Installation

```bash
python setup.py develop
```

See requirements.txt for our setup. We require Pytorch >= 1.8.0 and used the master versions of GPyTorch and BoTorch installed from source.

## File Structure

```
.
+-- volatilitygp/
|   +-- likelihoods/
|   |   +-- _one_dimensional_likelihood.py (Implementation of Newton iteration and the base class for the others)
|   |   +-- bernoulli_likelihood.py
|   |   +-- binomial_likelihood.py
|   |   +-- fixed_noise_gaussian_likelihood.py
|   |   +-- multivariate_normal_likelihood.py
|   |   +-- poisson_likelihood.py
|   |   +-- volatility_likelihood.py
|   +-- mlls/
|   |   +-- patched_variational_elbo.py (patched version of elbo to allow sumMLL training)
|   +-- models/
|   |   +-- model_list_gp.py (patched version of ModelListGP to allow for SVGP models)
|   |   +-- single_task_variational_gp.py (Our basic model class for SVGPs)
|   +-- utils/
|   |   +-- pivoted_cholesky.py (our pivoted cholesky implementation for inducing point init)
+-- experiments/
|   +-- active_learning/ (malaria experiment)
|   |   +-- qnIPV_experiment.py (main script)
|   +-- highd_bo/ (rover experiments)
|   |   +-- run_trbo.py (turbo script)
|   |   +-- run_gibbon.py (global model script, Fig 10c)
|   |   +-- rover_conditioning_experiment.ipynb (Fig 10b)
|   |   +-- trbo.py (turbo implementation)
|   +-- hotspots/ (schistomiasis experiment)
|   |   +-- hotspots.py (main script)
|   +-- mujoco/ (mujoco experiments on swimmer and hopper)
|   |   +-- functions/ (mujoco functions)
|   |   +-- lamcts/ (LA-MCTS implementation)
|   |   +-- turbo_1/ (TurBO implementation)
|   |   run.py (main script)
|   +-- pref_learning/ (preference learning experiment)
|   |   +-- run_pref_learning_exp.py (main script)
|   +-- std_bayesopt/ (bayes opt experiments)
|   |   +-- hartmann6.py (constrained hartmann6)
|   |   +-- lcls_optimization.py (laser)
|   |   +-- poisson_hartmann6.py (poisson constrained hartmann6)
|   |   +-- utils.py (model definition helpers)
|   |   +-- weighted_gp_benchmark/ (python 3 version of WOGP)
|   |   |   +-- lcls_opt_script.py (main script)
+-- tests/ (assorted unit tests for the volatilitygp package)
```

## Commands

Please see each experiment folder for the larger scale experiments. 

The understanding experiments can be found in:
- Figure 1a-b: `notebooks/svgp_fantasization_plotting.ipynb`
- Figure 1c: `notebooks/SABR_vol_plotting.ipynb`
- Figure 2b-d: `experiments/std_bayesopt/knowledge_gradient_branin_plotting.ipynb`
- Figure 6: `notebooks/ssgp_port.ipynb`
- Figure 7: `notebooks/ssgp_time_series_testing_pivcholesky.ipynb`
- Figure 8: `notebooks/streaming_bananas_plots.ipynb`
- Figure 10b: `experiments/highd_bo/rover_conditioning_experiment.ipynb`


## Code Credits and References

- BoTorch (https://botorch.org). Throughout, many examples were inspired by assorted BoTorch tutorials, while we directly compare to Botorch single task GPs.
- GPyTorch (https://gpytorch.ai). Our implementation of SVGPs rests on this implementation.
- LA-MCTS code comes from [here](https://github.com/facebookresearch/LaMCTS)
- laser WOGP code comes from [here](https://github.com/ermongroup/bayes-opt)
- hotspots data comes from [here](https://github.com/disarm-platform/adaptive_sampling_simulation_r_functions)
- malaria active learning script comes from [here](https://github.com/wjmaddox/online_gp). Data can be downloaded from [here](https://wjmaddox.github.io/data/https://wjmaddox.github.io/assets/data/malaria_df.hdf5).
