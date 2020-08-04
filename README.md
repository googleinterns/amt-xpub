# AMT Summer (2020) Intern Project --- Open Measurement

Fan Chen (fanci@)

Host: Jiayu Peng (jiayupeng@), Xichen Huang (huangxichen@, co-host)

## Overview

This repository includes code for

* Evaluation of the "shuffling+flipping" algorithm for bloom filter (BF). In this algorithm, the "shuffling" refers to a random permutation of bits in the BF, and the "flipping" refers to independent flipping (with a certain probability) of bits in the BF. We evaluate this algorithm in terms of its differential privacy (and accuracy). 

### Quickstart
It is recommended to use a virtual environment with this project. If you already
have one, you can skip to the next step.
The quickest way to set up a virtual environment is by running:
```
python3 -m venv env
source env/bin/activate
```
To install the requirements as well as the evaluation framework, simply run:
```
pip install -r requirements.txt
python setup.py install
```
After these steps, the code and its dependencies will be installed as Python packages.

### Example
```python
import buffling
```