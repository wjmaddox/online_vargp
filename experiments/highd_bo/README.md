## High Dimensional BO on Rover

The rover functions were copied over from https://github.com/zi-w/Ensemble-Bayesian-Optimization/ and then we ran 2to3 on them to convert to python 3.

An example model fit is in `rover_conditioning_experiment.ipynb`

### Command for TurBO scripts

```bash
python run_trbo.py --method={variational,exact,sgpr} --problem=rover \
    --batch_size=100 --num_init=200 --n_batch=200 \
    --num_inducing={NUM_INDUCING} --loss={LOSS} --acqf={ts,ts_rollout} --tree_depth={DEPTH} \
    --seed={SEED} --output={OUTPUT} > {OUTPUT.LOG}

# example to produce a run of Figure 5a,b
# other options are for ablations in appendix
python run_trbo.py --method=variational --problem=rover \
    --batch_size=100 --num_init=200 --n_batch=200 \
    --num_inducing=500 --loss=pll --acqf=ts_rollout --tree_depth=4 \
    --seed=2019 --output=rollout.pt > rollout.log
```

