## Command

```bash
python ./run_pref_learning_exp.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --method=variational \
        --output=$DIR/volatilitygp/pref_learning/plsim_variational_${SLURM_ARRAY_TASK_ID}.pt

python ./run_pref_learning_exp.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --method=laplace \
        --output=$DIR/volatilitygp/pref_learning/plsim_laplace_${SLURM_ARRAY_TASK_ID}.pt
```