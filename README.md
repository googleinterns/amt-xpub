# AMT Summer (2020) Intern Project --- Open Measurement

Fan Chen (fanci@)

Host: Jiayu Peng (jiayupeng@), Xichen Huang (huangxichen@, co-host)

## Overview

This repository includes code for evaluation of the "shuffling+flipping" algorithm for bloom filter (BF). 
`buffling` stands for "bit shuffling." 
In this algorithm, the "shuffling" refers to a random permutation of bits in the BF, and the "flipping" refers to independent flipping (with a certain probability) of bits in the BF. 
We evaluate this algorithm in terms of its differential privacy.

## Quickstart
It is recommended to use a virtual environment with this project.
If you already have one, you can skip to the next step.
The quickest way to set up a virtual environment is by running:
```bash
python3 -m venv env
source env/bin/activate
```
To install the requirements as well as the evaluation framework, simply run:
```bash
pip install -r requirements.txt
python setup.py install
```
After these steps, the code and its dependencies will be installed as Python packages.

## Examples
* The `signal_to_noise_ratio` module evaluates the signal-to-noise ratio of the output of "shuffling+flipping". Please check out examples/plot_signal_to_noise_ratio_analysis_results.py, which will generate three figures of the SNR analysis (the plots will be saved to docs/fig/). 
* The `privacy` module evaluates the privacy for one or multiple bloom filters. Please check out: examples/evaluate_privacy_of_shuffling_and_flipping.py, which runs two privacy evaluation for one bloom filter and for multiple bloom filters respectively (the result will be printed to screen).

Which will run the same experiments across multiple different estimation methods

## Documentation 

Please check out: docs/technical-details.pdf, which contains all the design details and the mathmatical derivations. 

## Miscelluar 

This repo also contains some accuracy evaluation for any distribution bloom filter (ADBF). We compare exponential/geometric/logarithm BF. `docs/simulation-design.pdf` documents the evaluation designs and results. 
Under the `adbf_accuracy/` directory:

* `src/` contains the python/R scripts that run different simulations and plot for the results. To run the python code, you will need to install the world federation of advertisers (WFA) cardinality_estimation_evaluation_framework, which is available at [GitHub](https://github.com/world-federation-of-advertisers/cardinality_estimation_evaluation_framework). We tested the code using the commit version ``. You could install this version by checking out to this version: 
`git checkout bbc65a7f40d1d3e9c1036422d5646d1b995abbc9`
* `dat/` contains all the intermediate data output from the simulations. 
* `fig/` contains the resulting plots, including exploratory data analysis and model diagnostics. 
