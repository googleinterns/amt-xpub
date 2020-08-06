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


"""Evaluate differential privacy of the 'shuffling+flipping' algorithm. 

* scientific computation: 
  * evaluate_gauss_continued_fraction() and 
  * ratio_of_hypergeometric_mgf()
* DP evaluation for one bloom filter: 
  * evaluate_eps_given_yd()
  * find_yx_given_yd_Is_1()
  * find_p_given_yd_is_1
* Evaluation for multiple bloom filters:
  * estimate_eps() 
  * estimate_p() 
"""


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
  count_input_ones, p, n_bit, type="exact", 
  count_output_ones=None, depth=1000):
  """Evaluate the privacy parameter of given count of input ones.

  Evaluate the privacy of one bloom filter given the count of input ones. 

  Args: 
    count_input_ones: An integer, the # of 1s in the input bloom filter.
    p: An real number in [0,1], bit flipping probability.
    n_bit: Integer, the length of each bloom filter.
    type: A character string, either "exact" or "asymp".
    count_output_ones: An integer, the # of 1s in the output bloom filter.
    depth: Integer, Approximating depth. 
      Will be Passed to evaluate_gauss_continued_fraction().

  Returns:
    A real number of the simulated epsilon.
  """
  ## process the inputs
  q = 1 - p
  if count_output_ones is None:
    count_output_ones = np.random.binomial(count_input_ones, q) + \
      np.random.binomial(n_bit - count_input_ones, p)
    if count_input_ones + count_output_ones > n_bit: 
      count_input_ones = n_bit - count_input_ones
      count_output_ones = n_bit - count_output_ones

  if (type == "asymp"): ## normal approximation
    b = 2 * np.log(q / p)
    l = -b * count_input_ones / n_bit + \
      b ** 2 / 2 * count_input_ones * (n_bit - count_input_ones) * \
      (2 * count_output_ones - n_bit) / n_bit ** 3
    l = np.exp(l)
  elif (type == "exact"): ## exact form
    l = ratio_of_hypergeometric_mgf(
      count_input_ones=count_input_ones, 
      count_output_ones=count_output_ones, 
      p=p, n_bit=n_bit, depth=depth
    ) 
  else:
    raise ValueError("type should be either `asympt` or `exact`.")
  
  ## Pr(A(D')=X) / Pr(A(D)=X)
  privacy_loss = p / q / l ## version 2

  return np.log(privacy_loss)


def find_quantile_of_count_of_output_ones(p, n_bit, d):
  '''Find a quantile of the count of 1s in one output bloom filter.
  
  Consider one bloom filter and assume the input bloom filter has 1 one. 
  This funciton finds the the d quantile of the count of ones in the output BF. 
  Note: The count of ones in the output is a random variable. 
  
  Args: 
    p: An real number in [0,1], bit flipping probability.
    n_bit: Integer, the length of each bloom filter.
    d: A real number, the approximation parameter.
  
  Returns: 
    An integer, the # of 1s in bloom filter after "shuffling+blipping".
  '''
  q = 1 - p
  v = stats.binom(n_bit - 1, p) ## pmf of count_output_ones
  lower = 0
  upper = n_bit
  count_output_ones = (lower + upper) // 2
  while lower + 1 < upper: 
    cdf_output_ones = p * v.cdf(count_output_ones) + \
      q * v.cdf(count_output_ones - 1) 
    if cdf_output_ones > d:
      upper = count_output_ones
    else: 
      lower = count_output_ones
    count_output_ones = (lower + upper) // 2
  
  return count_output_ones


def find_flip_prob_of_one_bloom_filter(e, n_bit, d=1e-4):
  """Find the needed flipping probability for one bloom filter. 
  
  Find the flipping probability (p) needed for a target privacy parameter. 
  Use binary search for p. The privacy parameter is evaluated by assuming 
  exactly one observed 1.

  Args: 
    e: A real number, the privacy parameter (epsilon).
    n_bit: Integer, the length of each bloom filter.
    d: A real number, the approximation parameter.

  Returns:
    A real number of the estimated p.
  """
  
  lower = 0
  upper = .5
  p = (lower + upper) / 2
  while lower + 1e-7 < upper:
    epsilon = np.zeros(2)

    ## find d/2 quantile
    count_output_ones = find_quantile_of_count_of_output_ones(p, n_bit, d / 2)
    epsilon[0] = evaluate_privacy_of_one_bloom_filter(
      1, p, n_bit, depth=1000, count_output_ones=count_output_ones)
    
    ## find 1-d/2 quantile
    count_output_ones = find_quantile_of_count_of_output_ones(
      p, n_bit, 1 - d / 2)
    epsilon[1] = evaluate_privacy_of_one_bloom_filter(
      1, p, n_bit, depth=1000, count_output_ones=count_output_ones)

    epsilon_hat = np.max(abs(epsilon))
    if epsilon_hat > e:
      lower = p
    else: 
      upper = p
    p = (lower + upper) / 2

  return p


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