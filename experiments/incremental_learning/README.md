This is a monkey patched version of the online gp codebase from Stanton et al, '21 available at 
https://github.com/wjmaddox/online_gp .

We modified it to use pivoted cholesky updates.
## Command

```bash
runSplits () {
        for split in 0 1 2 3 4 5 6 7 8 9;
        do
                python experiments/regression.py model=$1 dataset=$2 logging_freq=400 dtype=$3 \
                        dataset.dataset_dir=$DIR/online_gp_private/data/uci/$2/ \
                        dataset.split_seed=$split stem=eye
        done
}

runSplits pc_svgp_regression protein float64
runSplits svgp_regression protein float 64
```
