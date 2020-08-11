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


"""Evaluate the privacy for one/multiple bloom filter(s)."""


from absl import app
from absl import flags
import buffling.privacy as bp
import numpy as np
import pandas as pd


# global parameters
FLAGS = flags.FLAGS
flags.DEFINE_float("epsilon", np.log(3), "Privacy parameter (multi-BFs)")
flags.DEFINE_float("delta", 1e-4, "Approximation parameter (multi-BFs)")
flags.DEFINE_integer("n_pub", 6, "Number of publishers") 
flags.DEFINE_integer("n_bit", 100000, "Number of bits (sketch size)") 
flags.DEFINE_integer("n_simu", 100000, "Number of sampling times") 


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # required p for one BF
  data = pd.DataFrame(columns=['# input 1s', 'epsilon'])
  for count_input_ones in (1, 12509, 37509, 62509, 87509, 99999):
    p_hat = bp.evaluate_privacy_for_one_bloom_filter(
      count_input_ones=count_input_ones, p=0.05, n_bit=FLAGS.n_bit, 
      d=0.001, n_simu=5000)
    data.loc[len(data)] = [count_input_ones, p_hat]
  print("One bloom filter of {FLAGS.n_bit} bits.")
  print("Estimated privacy parameter (delta=0.001):")
  print(data)

  # required p for multiple BFs
  print(f"\n{FLAGS.n_pub} publishers, {FLAGS.n_bit} bits")
  print(f"epsilon = {FLAGS.epsilon}, \
  delta = {FLAGS.delta}, sampling {FLAGS.n_simu} times")
  p = bp.estimate_flip_prob(
    FLAGS.n_pub, FLAGS.n_bit, e=FLAGS.epsilon, 
    d=FLAGS.delta, n_simu=FLAGS.n_simu) 
  print(f"Required flipping probability = {p}.")


if __name__ == '__main__':
  app.run(main) 
