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


from buffling import privacy
import numpy as np
import pandas as pd


# global parameters
EPSILON = np.log(3)
DELTA = 1e-4
NUM_PUB = 8
NUM_BIT = 1000000
NUM_SAMPLING = 500000


def main():
  # required p for one BF
  print(f"epsilon = {EPSILON}, delta = {DELTA}\n")
  data = pd.DataFrame(columns=['BF size', 'required p'])
  for n_bit in (1000, 2000, 5000, 10000, 20000, 50000, 
                100000, 200000, 500000, 1000000, 2000000):
    p_hat = privacy.find_p_given_yd_is_1(e=EPSILON, n_bit = n_bit, d=DELTA)
    data.loc[len(data)] = [n_bit, p_hat]
  data = data.astype({'BF size': 'int'})
  print("\n", data)


  # required p for multiple BFs
  print(f"\n{NUM_PUB} publishers, {NUM_BIT} bits")
  print(f"epsilon = {EPSILON}, delta = {DELTA}, sampling {NUM_SAMPLING} times")
  p = privacy.estimate_p(NUM_PUB, NUM_BIT, e=EPSILON, d=DELTA, n_simu=NUM_SAMPLING) 
  print(f"Required flipping probability = {p}.")


if __name__ == '__main__':
   main() 
