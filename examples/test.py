# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import buffling.privacy as bp
import numpy as np
import pandas as pd


# global parameters
EPSILON = np.log(3)
DELTA = 1e-4
NUM_PUB = 1
NUM_BIT = 10000
NUM_SAMPLING = 100000


def main():
  # required p for one BF
  data = pd.DataFrame(columns=['# input 1s', 'epsilon'])
  for count_input_ones in (1, 1250, 3750, 6250, 8750, 9999):
    p_hat = bp.evaluate_privacy_of_one_bloom_filter(
      count_input_ones=count_input_ones, p=0.05, n_bit=NUM_BIT, 
      d=0.001, n_simu=5000)
    data.loc[len(data)] = [count_input_ones, p_hat]
  print("One bloom filter of {NUM_BIT} bits.")
  print("Estimated privacy parameter (delta=0.001):")
  print(data)

  # required p for multiple BFs
  print(f"\n{NUM_PUB} publishers, {NUM_BIT} bits")
  print(f"epsilon = {EPSILON}, delta = {DELTA}, sampling {NUM_SAMPLING} times")
  p = bp.estimate_flip_prob(
    NUM_PUB, NUM_BIT, e=EPSILON, d=DELTA, n_simu=NUM_SAMPLING) 
  print(f"Required flipping probability = {p}.")


if __name__ == '__main__':
   main() 
