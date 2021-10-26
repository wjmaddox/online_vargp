## Commands

```bash
python hotspots.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --batch_limit=8 --num_init=100 \
    --beta=0.1 --loss=elbo --dataset=civ --inner_samples=16 --outer_samples=16 \
    --output=$DIR/volatilitygp/hotspots_fp/civ_ind_svgp_${SLURM_ARRAY_TASK_ID}_AMD.pt

python hotspots.py --seed=$SLURM_ARRAY_TASK_ID --n_batch=100 --num_init=100 \
    --beta=0.1 --loss=elbo --dataset=civ --random \
    --output=$DIR/volatilitygp/hotspots_fp/civ_ind_svgp_${SLURM_ARRAY_TASK_ID}_AMD.pt
```