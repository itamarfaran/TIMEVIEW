# Get the argument --debug to run the experiments in debug mode. This will run the experiments with a smaller number of trials and tune iterations.
# Example: bash Table_3.sh --debug

# Check if --debug is in the arguments and set the number of trials and tune iterations accordingly.
if [[ " $@ " =~ " --debug " ]]; then
    n_trials=1
    n_tune=1
else
    n_trials=10
    n_tune=100
fi


# Comment this part if you already have run TIMEVIEW_interface_only.sh
if [[ " $@ " =~ " --debug " ]]; then
    bash ./run_scripts/TIMEVIEW_interface_only.sh --debug
else
    bash ./run_scripts/TIMEVIEW_interface_only.sh
fi

python benchmark.py --datasets airfoil_log flchain_1000 stress-strain-lot-max-0.2 synthetic_tumor_wilkerson_1 sine_trans_200_20 beta_900_20 --baselines TTS --n_trials $n_trials --n_tune $n_tune --seed 0 --device gpu --n_basis 9
