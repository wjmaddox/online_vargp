## Commands

```bash
python qnIPV_experiment.py --seed=$SLURM_ARRAY_TASK_ID --num_init=10 --model=svgp \
        --num_steps=250 --seed=${SLURM_ARRAY_TASK_ID} \
        --output=$DIR/volatilitygp/malaria/malaria_nipv_svgp_${SLURM_ARRAY_TASK_ID}.pt

python qnIPV_experiment.py --seed=$SLURM_ARRAY_TASK_ID --num_init=10 --model=svgp \
        --cuda --num_steps=250 --random --seed=${SLURM_ARRAY_TASK_ID} \
        --output=$DIR/volatilitygp/malaria/malaria_random_svgp_${SLURM_ARRAY_TASK_ID}.pt

python qnIPV_experiment.py --seed=$SLURM_ARRAY_TASK_ID --num_init=10 --model=exact \
        --num_steps=250 --seed=${SLURM_ARRAY_TASK_ID} \
        --output=$DIR/volatilitygp/malaria/malaria_nipv_svgp_${SLURM_ARRAY_TASK_ID}.pt

python qnIPV_experiment.py --seed=$SLURM_ARRAY_TASK_ID --num_init=10 --model=exact \
        --cuda --num_steps=250 --random --seed=${SLURM_ARRAY_TASK_ID} \
        --output=$DIR/volatilitygp/malaria/malaria_random_svgp_${SLURM_ARRAY_TASK_ID}.pt
```