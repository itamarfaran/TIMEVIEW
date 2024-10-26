# Get the argument --debug to run the experiments in debug mode. This will run the experiments with a smaller number of trials and tune iterations.
# Example: bash Table_3.sh --debug

# Check if --debug is in the arguments and set the number of trials and tune iterations accordingly.
if [[ " $@ " =~ " --debug " ]]; then
    n_trials=1
    n_tune=1
else
    n_trials=25
    n_tune=100
fi


python benchmark.py --datasets airfoil_log flchain_1000 stress-strain-lot-max-0.2 synthetic_tumor_wilkerson_1 --baselines TTS --n_trials $n_trials --n_tune $n_tune --seed 0 --n_basis 9  --device cpu
python benchmark.py --datasets sine_trans_200_20 beta_900_20 --baselines TTS --n_trials $n_trials --n_tune $n_tune --seed 0 --n_basis 5  --device cpu
