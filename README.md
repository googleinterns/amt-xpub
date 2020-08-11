# AMT Summer (2020) Intern Project --- Open Measurement

Fan Chen (fanci@)

Host: Jiayu Peng (jiayupeng@), Xichen Huang (huangxichen@, co-host)

## Overview

This repository includes code for evaluation of the "shuffling+flipping" algorithm for bloom filter (BF). 
In this algorithm, the "shuffling" refers to a random permutation of bits in the BF, and the "flipping" refers to independent flipping (with a certain probability) of bits in the BF. 
We evaluate this algorithm in terms of its differential privacy (and accuracy). 

### Quickstart
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
After these steps, the code and its dependencies will be installed as Python 
packages.

### Example
`buffling` stands for "bit shuffling." 
To start with and get a sense for how this code all works together, check out the examples/ folder.

Which will run the same experiments across multiple different estimation methods

### Documentation 

Please refer to `docs/technical-details.pdf` for more details of the design and the mathmatical derivations. 

#### Signal-to-noise ratio (`signal_to_noise_ratio` module)
The `signal_to_noise_ratio` module contains `calculate_signal_to_noise_ratio()` which evaluates the signal-to-noise ratio of the output of "shuffling+flipping".

#### Privacy evaluation (`privacy` module)

The `privacy` module contains the five functions.

* For scientific computating (helpers): 
  * `evaluate_gauss_continued_fraction`()  
  * `ratio_of_hypergeometric_mgf`()
* For privacy evaluation for bloom filter(s):
  * `evaluate_privacy_of_one_bloom_filter`()
  * `estimate_privacy_of_bloom_filter`() 
  * `estimate_flip_prob`() 