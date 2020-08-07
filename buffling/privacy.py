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


"""Evaluate differential privacy of the 'shuffling+flipping' algorithm."""


import numpy as np
import scipy.stats as stats


def evaluate_gauss_continued_fraction(a, b ,c, z, depth=1000):
  """Approximate the value of Gauss' continued fraction. 
  
  Calculate phi(a + 1, b; c + 1; z) / phi(a, b; c; z)
  where phi is the hypergeometric function.

  Args: 
    a, b, c, z: Numeric parameters to the function. 
    depth: Positive integer, Approximating depth.

  Returns:
    The approximated value.
  """
  k = np.zeros(depth)
  for i in range(depth): 
    if i % 2 == 0:
      variable = i // 2
      k[i] = (a - c - variable) * (b + variable) 
    else: 
      variable = i // 2 + 1
      k[i] = (b - c - variable) * (a + variable)
    if c == 0 and i == 0:
      k[0] /= 1
    else: 
      k[i] /= (c + i) * (c + i + 1)
  f = np.ones(depth)
  for i in range(depth - 1):
    f[depth - i - 2] += k[depth - i - 2] * z / f[depth - i - 1]
  return 1 / f[0]


def ratio_of_hypergeometric_mgf(
  count_input_ones, count_output_ones, p, n_bit, depth=1000):
  """Compute the ratio of two hypergeometric moment-generating functions.

  Definition: E((q/p)^(2x)) / E((q/p)^(2y)),
  where x ~ HG(m, count_input_ones, count_output_ones), and 
  y ~ HG(m, count_input_ones + 1, count_output_ones) 
  Note: The increment change means that we have count_input_ones + 1.

  Args: 
    count_input_ones: An integer, the # of 1s in the input bloom filter.
    count_output_ones: An integer, the # of 1s in the output bloom filter.
    p: An real number in [0,1], bit flipping probability.
    n_bit: Integer, the length of each bloom filter.
    depth: Positive integer, Approximating depth. 
      Will be Passed to evaluate_gauss_continued_fraction().

  Returns:
    A real number of the simulated epsilon.
  """
  q = 1 - p
  if count_input_ones + count_output_ones > n_bit: 
    count_input_ones = n_bit - count_input_ones
    count_output_ones = n_bit - count_output_ones
  
  a = - count_input_ones
  b = - count_output_ones
  c = n_bit - count_output_ones - count_input_ones 
  z = (q / p) ** 2
  l = evaluate_gauss_continued_fraction(a, b, c, z, depth=depth)

  return (n_bit - count_output_ones) / \
    np.maximum(n_bit - count_output_ones - count_input_ones, 1) * l


def evaluate_privacy_of_one_bloom_filter(
  count_input_ones, p, n_bit, d=1e-4, n_simu=50000, depth=1000):
  """Evaluate the privacy of one bloom filter.

  Evaluate the privacy of one bloom filter given the count of input ones.

  Args: 
    count_input_ones: Integer, the # of 1s in the input bloom filter.
    p: An real number in [0,1], bit flipping probability.
    n_bit: Integer, the length of each bloom filter.
    d: A real number, the approximation parameter.
    n_simu: Integer, the number of sampling times. Shoule be larger than 1/d.
    depth: Integer, Approximating depth. 
      Will be Passed to evaluate_gauss_continued_fraction().

  Returns:
    A real number of the simulated epsilon.
  """
  ## process the inputs
  q = 1 - p
  if isinstance(count_input_ones, int):
    count_input_ones = np.repeat(count_input_ones, n_simu)

  ## sample count_output_ones, given count_input_ones
  count_output_ones = np.random.binomial(count_input_ones, q, n_simu) + \
    np.random.binomial(n_bit - count_input_ones, p, n_simu)
  r = np.zeros(n_simu)
  for i in range(n_simu):
    r[i] = ratio_of_hypergeometric_mgf(
      count_input_ones[i], count_output_ones[i], 
      p=p, n_bit=n_bit, depth=depth) 
  privacy_loss = p / q / r ## Pr(A(D')=X) / Pr(A(D)=X)
  e = np.quantile(np.abs(np.log(privacy_loss)), 1 - d)

  return e


def estimate_privacy_of_bloom_filter(
  n_pub, n_bit, p, d=1e-4, n_simu=100000):
  """Calculate the privacy parameter (epsilon) of bloom filters. 

  Args: 
    n_pub: Integer, the number of publishers.
    n_bit: Integer, the length of each bloom filter.
    p: A real number in (0,1), the flipping probability.
    d: A real number, the approximation parameter.
    n_simu: Integer, the number of sampling times. 
      This shoule be larger than 1/d.

  Returns:
    An estimated of flipping probability.
  """
  q = 1 - p
  P0 = [stats.binom.pmf(y, n_pub, p) for y in range(n_pub+1)]
  Pk = [stats.binom.pmf(y, n_pub, q) for y in range(n_pub+1)]

  ## g|D', take 1-delta quantile of ln(R)
  g0 = stats.multinomial.rvs(n_bit-1, P0, size=n_simu)
  gk = stats.multinomial.rvs(1, Pk, size=n_simu)
  g = g0 + gk
  w = [(q / p) ** (2 * y - n_pub) for y in range(n_pub+1)]
  privacy_loss = g.dot(w) / n_bit
  e0 = np.quantile(np.log(privacy_loss), 1 - d)

  ## g|D, take delta quantile of ln(R)
  g = stats.multinomial.rvs(n_bit, P0, size=n_simu)
  w = [(q / p) ** (2 * y - n_pub) for y in range(n_pub+1)]
  privacy_loss = g.dot(w) / n_bit
  e1 = np.quantile(-np.log(privacy_loss), 1 - d)

  return max(e0, e1)


def estimate_flip_prob(
  n_pub, n_bit, e=np.log(3), d=1e-4, n_simu=100000, tol=1e-5):
  """Search for needed flipping probability (p). 

  Args: 
    n_pub: Integer, the number of publishers.
    n_bit: Integer, the length of each bloom filter.
    e: target privacy parameter (epsilon).
    d: A real number, the approximation parameter.
    n_simu: Integer, the number of sampling times. Shoule be larger than 1/d.
    tol: A small real number, the tolerance of binary search. 

  Returns:
    An estimated of flipping probability.
  """
  if tol <= 0:
    raise ValueError('"tol" should be positive.')
  lower = 0
  upper = 0.5
  p = (lower + upper) / 2
  while lower + tol < upper:
    e_hat = estimate_privacy_of_bloom_filter(
      n_pub, n_bit, p, d=d, n_simu=n_simu)
    if e_hat < e:
      upper = p
    else: 
      lower = p
    p = (lower + upper) / 2
  return p