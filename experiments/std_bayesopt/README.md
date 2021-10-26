## Commands

### Constrained Hartmann-6

```bash
python ./hartmann6.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=150 --method=variational \
        --output=$DIR/volatilitygp/hartmann6/hartmann6_fixednoise3_250_kg_variational_${SLURM_ARRAY_TASK_ID}.pt

python ./hartmann6.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=150 --method=exact \
       --output=$DIR/volatilitygp/hartmann6/hartmann6_fixednoise2_250_kg_exact_${SLURM_ARRAY_TASK_ID}.pt
```

### Poisson Constrained Hartmann-6

```bash
python ./poisson_hartmann6.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=150 --method=variational \
        --output=$DIR/volatilitygp/hartmann6/poisson_hartmann6_fixednoise3_250_kg_variational_${SLURM_ARRAY_TASK_ID}.pt

python ./hartmann6.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=150 --method=exact \
     --output=$DIR/volatilitygp/hartmann6/hartmann6_fixednoise2_250_kg_exact_${SLURM_ARRAY_TASK_ID}.pt
```

### Laser

```bash
python ./lcls_optimization.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --method=variational \
        --batch_size=1 \
        --output=$DIR/volatilitygp/laser/laser_100_kg_variational_30_${SLURM_ARRAY_TASK_ID}.pt

python ./lcls_optimization.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --method=exact \
         --batch_size=1 \
      --output=$DIR/volatilitygp/laser/laser_100_kg_exact_${SLURM_ARRAY_TASK_ID}.pt

python lcls_opt_script.py # in the weighted_gps_benchmark for the WOGP baseline
```

