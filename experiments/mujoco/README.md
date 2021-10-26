## Commands

### Swimmer
```bash
python run.py --func=swimmer --iterations=50 --method=exact --acqf=ts_rollout \
      --dirname=$DIR/volatilitygp/swimmer_trial/exact_tsr2_trial_${SLURM_ARRAY_TASK_ID}/ > \
        $DIR/volatilitygp/swimmer_trial/exact_tsr2_trial_${SLURM_ARRAY_TASK_ID}.log

python run.py --func=swimmer --iterations=50 --method=variational --acqf=ts_rollout \
        --dirname=$DIR/volatilitygp/swimmer_trial/var_tsr2_trial_${SLURM_ARRAY_TASK_ID}/ > \
        $DIR/volatilitygp/swimmer_trial/var_tsr2_trial_${SLURM_ARRAY_TASK_ID}.log

python run.py --func=swimmer --iterations=50 --method=exact --acqf=ts \
        --dirname=$DIR/volatilitygp/swimmer_trial/exact_ts_trial_${SLURM_ARRAY_TASK_ID} > \
        $DIR/volatilitygp/swimmer_trial/exact_ts_trial_${SLURM_ARRAY_TASK_ID}.log

python run.py --func=swimmer --iterations=50 --method=variational --acqf=ts \
        --dirname=$DIR/volatilitygp/swimmer_trial/variational_ts_trial_${SLURM_ARRAY_TASK_ID} > \
        $DIR/volatilitygp/swimmer_trial/variational_ts_trial_${SLURM_ARRAY_TASK_ID}.log
```

### Hopper

```bash
python run.py --func=hopper --iterations=170 --method=${method} --acqf=${acqf} --seed=${seed} \
        --dirname=$DIR/volatilitygp/hopper_trial/${method}_${acqf}_trial_${seed} > \
        $DIR/volatilitygp/hopper_trial/${method}_${acqf}_trial_${seed}.log
```


# Latent Action Monte Carlo Tree Search (LA-MCTS)
LA-MCTS is a meta-algortihm that partitions the search space for black-box optimizations. LA-MCTS progressively learns to partition and explores promising regions in the search space, so that solvers such as Bayesian Optimizations (BO) can focus on promising subregions, mitigating the over-exploring issue in high-dimensional problems. 

<p align="center">
<img src='https://github.com/linnanwang/paper-image-repo/blob/master/LA-MCTS/meta_algorithms.png?raw=true' width="300">
</p>

Please reference the following publication when using this package. ArXiv <a href="https://arxiv.org/abs/2007.00708">link</a>.

```
@article{wang2020learning,
  title={Learning Search Space Partition for Black-box Optimization using Monte Carlo Tree Search},
  author={Wang, Linnan and Fonseca, Rodrigo and Tian, Yuandong},
  journal={NeurIPS},
  year={2020}
}
```
