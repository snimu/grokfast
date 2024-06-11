# hlb-gpt-cli

CLI controllable version of hlb-gpt by tysam-code

[Fern](https://github.com/tysam-code)'s [hlb-gpt](https://github.com/tysam-code/hlb-gpt)
is a package for training transformers really, really fast, to high quality.

This package extends that one that one to be easily controllable over the CLI. It also adds extensive logging.

## Why does this exist?

1. It makes ablations really easy
2. It gives you extremely detailed logs

Just provide a whole bunch of settings over the command-line, and automatically run highly repeatable experiments at different model scales, and log their results in high detail
to either wandb and/or a local .csv-file.

## Example

How about we ablate the number of attention heads over different widths and depths?
You could run:

```bash
python v041.py -cw --wandb_project test --seed 1000 --num_runs 5 --max_epochs 5 --depth 4 8 16 32 --width 192 384 --num_heads 1 3 6
```

What does that do?

- `-cw`: Equivalent to `--log_csv --log_wandb`. Log both to a .csv-file (by default, "results_041.csv"), and to a wandb-project.
- `--wandb_project test`: Log to the wandb-project "test"
- `--seed 1000`: Manually set the seed. 
- `--num_runs 5`: For each setting, do train 5 times. Each run is initialized with a different seed, starting with 
    the one given in `--seed`. 
    So here, in each settings, the corresponding 5 runs would have the seeds [1000, 1001, 1002, 1003, 1004],
    making them highly comparable and repeatable.
- `--max_epochs 5`: Training runs for 5 epochs. An eval is guaranteed at the end.
- `--depth 4 8 16 --width 192 384 --num_heads 1 3 6`: Every combination of these values represents one setting, and will be run for 5 runs.
    If `width % num_runs != 0` in some settings, that setting won't be run.
    The idea of the package is that you can extend the code to add whatever change you want as a setting,
    and easily test it together with several other setups.

As you can see, it is easy to run very large and highly repeatable ablations of different settings.
I tried to make the code easy to extend, so that you can easily ablate your own settings.
More on that below.


**Limitation**

There are several versions of hlb-gpt. The ones that I found relevant are v0.3.0, which implements a classic
transformer with subsequent Attention and MLP layers, and v0.4.1, which implements a fused
Attention and MLP architecture, where GELU is applied to the values in Attention.

So far, I only implemented a CLI interfacte to the latter, under v041.py, though I do plan to write a v030.py script, too.


## Existing args

Here are the CLI-args that you can currently use to run this script.

- **-c, --log_csv** (FLAG) 
    - If set, log results to .csv-file.
- **--append** (FLAG) 
    - If set, the previous logfile won't be overwritten but appended to, if it already exists.
- **--logfile** (TYPE: str, DEFAULT: 'results_041.csv')
    - Log the results to this file.
- **-w, --log_wandb** (FLAG) 
    - If set, log results to Weights & Biases.
- **--wandb_project** (TYPE: str, DEFAULT: 'speedy-lang') 
    - Weights & Biases project to log to.
- **--review_settings** (FLAG)
    - Print the settings before proceeding to review them.
    - Useful because some settings might be pre-filtered
        (for example, if you have different widths and num_heads,
        only the combinations where width is divisible by num_heads are used).
        If something is wrong with the settings, you can easily see it here, return early, and fix it.
- **--num_runs** (TYPE: int, DEFAULT: 1)
    - Number of times to run each experiment for.
    - Each run for a single setting will start with a different seed,
        but over the different settings, the seeds are repeated run-by-run to get comparable results.
    - Increase this to get more statistically significant results.
- **--max_epochs** (TYPE: int, DEFAULT: 1)
    - If epoch>=max_epochs, stop training and eval one last time.
    - By default, this is the determining factor for training length.
    - This way, you can be sure that your model was trained on the entire dataset, and can quantify data repetition.
    - At some point, more datasets might be supported, in which case this makes for the most flexible setting.
- **--max_steps** (TYPE: int, DEFAULT: int(1e9))
    - If step>=max_steps, stop training and eval one last time.
    - Very high by default so that epochs are the determining factor by default.
    - One step does *not* correspond to a constant number of tokens, as the batch size and sequence length are adjusted dynamically. 
    - Therefore, this is a measure of limited utility, and is mostly included for the sake of completeness.
- **--max_tokens** (TYPE: int, DEFAULT: int(1e12))
    - If token>=max_tokens, stop training and eval one last time.
    - Very high by default so that epochs are the determining factor by default.
    - Due to the automatically adjusted batchsize and sequence length,
        you are not guaranteed to get a consistent number of tokens used in training this way.
- **--max_time_seconds** (TYPE: int, DEFAULT: int(1e9))
    - If t_secs>=max_time_seconds, stop training and eval one last time.
    - Very high by default so that epochs are the determining factor by default.
    - Due to the automatically adjusted batchsize and sequence length,
        the actual time that the training stops after may vary over the runs.
- **--max_epochs_between_evals** (TYPE: float, DEFAULT: 0.25)
    - Eval after at most this many epochs.
    - By default, only do a full eval every 50 steps (and a partial one on training data every 10 steps).
    - But over training, the number of tokens seen per step increases, which can mean several epochs between evals.
    - This parameter makes sure that you have enough eval data.
- **--model_scale** (TYPE: float, DEFAULT: 1.0, NARGS: '+')
    - Scale the model size. 
    - Can be overwritten by setting depth and width. 
    - You can provide multiple values to test multiple scales.
    - After scaling, the width will be rounded to the nearest multiple of 64,
        meaning that the true scale may differ slightly from the one provided here.
    - The true model scale will, however be calculated and logged.
- **--depth** (TYPE: int, DEFAULT: -1, NARGS: '+')
    - Depth of the model.
    - If <1, will be automatically determined via model_scale.
    - You can provide multiple values to test multiple depths.
    - If you set depth >= 1, you also have to set width to >= 1.
- **--width** (TYPE: int, DEFAULT: -1, NARGS: '+')
    - Width of the model.
    - If <1, will be automatically determined via model_scale.
    - Will always be rounded to the nearest multiple of 64.
    - You can provide multiple values to test multiple widths.
    - If you set width >= 1, you also have to set depth to >= 1.
- **--num_heads** (TYPE: int, DEFAULT: 1, NARGS: '+')
    - Number of attention heads.
    - The original implementation is single-headed, but this might prove valuable for some experiments.
    - You can provide multiple values to test multiple numbers of heads.
    - Only settings in which `width % num_heads == 0` are ever run.
- **--linear_value** (TYPE: int, DEFAULT: 0, NARGS: '+')
    - If 0, use Gelu on the value in attention (the default setting of this package), else don't.
    - If you provide several values (for example, 0 1 2 3 4), will be reduced to their booleans without repetition (so False, True).
    - TODO: make this bool? More typing but clearer.
- **--gpu_capacity_scalar** (TYPE: float, DEFAULT: 1.0)
    - Determines the combination of maximum sequence length and maximum batchsize to control max GPU memory usage.
    - 1.0 is for a 40GB A100; reduce or increase as needed. 
    - You may need to include some slack.
    - The adaptation is pretty good, but not perfect.
- **--seed** (TYPE: int, DEFAULT: 100)
    - Seed for the random number generator.
    - This determines the initial seed per experiment.
    - At each run, 1 is added to the seed, until the next setting.
    - For example: you have two settings and 3 runs each, with an initial seed of 100.
        Then the seeds for the 3 runs of setting 1 will be [100, 101, 102],
        and the seeds for the 3 runs of setting 2 will be identical to make them comparable.


## What is being logged?

By default, the following things are being logged:

*Config stuff:*

- **model_scale**: The scale of the model relative to the default of width=384 and depth=8
- **depth**: The number of transformer blocks in the network
- **width**: The width of the residual stream
- **num_params**: The total number of parameters in the model
- **num_non_embedding_params**: The number of non-embedding parameters
- **num_heads**: The number of attention heads
- **linear_value**: Whether or not linear values are used in the run
- **seed**: The actual seed for each run
- **run_num**: The run number

*Results:*

- **train_loss**: The train losses
- **val_loss**: The validation losses
- **train_acc**: The training accuracies
- **val_acc**: The validation accuracies
- **train_pplx**: The training perplexity
- **val_pplx**: The validation perplexity
- **grad_norm**: The L2-norm of the model gradients. Each entry corresponds to one entry in training, not validation
- **cumulative_time_train**: Cumulative time for each recorded training step
- **cumulative_time_val**: Cumulative time for each recorded validation step
- **tokens_seen_train**: Cumulative number of tokens seen during training
- **tokens_seen_val**: Cumulative number of tokens seen during validation
- **epochs_train**: The fractional epoch at each recorded training step
- **epochs_val**: The fractional epoch at each recorded validation step
- **batch_sizes_train**: The batch size at each recorded training step; recorded because it is dynamically adjusted
- **batch_sizes_val**: The batch size at each recorded validation step
- **seq_lengths_train**: The sequence length at each recorded training step
- **seq_lengths_val**: The sequence length at each recorded validation step
- **lrs_train**: The learning rates corresponding to each recorded training step; 
    different params have different learning rates.
    Their relative scales are recorded in hyp and not touched, so you can calculate the actual lr from them.
    TOOD: automatically record the learning rates for the different parameters
- **lrs_val**: The learning rates corresponding to each recorded validation step
- **weight_decays_train**: The weight decays corresponding to each recorded training step
- **weight_decays_val**: The weight decays corresponding to each recorded validation step

## How to extend the code

Will write more about this after some refactoring to make this easier. TODO

## Plotting the .csv

I have provided some utilities for plotting the results in the .csv-files in *plot_results.py*, if you want those.

There is also an `example_plot_fct`-function, so that you can see how the utils are meant to work together.
It provides great flexibility in plotting your sweeps.

## Plans

- [ ] Make the dataloader independent of the dataset, or at least easily changeable
- [ ] Improve the CLI interface based on community feedback, if any is incoming
- [ ] Reformat the code to make it more hackable
- [ ] Write v030.py, with an interface that is consistent with that of v041.py
- [x] Make plotting utils for the .csv-files available

## Citation

As already mentioned, the code is based on [Fern](https://github.com/tysam-code)'s [hlb-gpt](https://github.com/tysam-code/hlb-gpt):

```
cff-version: 1.2.0
message: "Citations would be appreciated if you end up using this tool! I currently go by Fern, no last name given."
authors:
  given-names: "Fern"
title: "hlb-gpt"
version: 0.4.1
date-released: 2023-03-05
url: "https://github.com/tysam-code/hlb-gpt"
```

If you want to cite me as well, just cite this repo and my name & GitHub handle.
