import numpy as np
import scipy.stats as stats

def GaussContFrac(a, b ,c, z, depth=1000):
  """
  Approximate the Gauss' continued fraction. 
  Calculate phi(a + 1, b; c + 1; z) / phi(a, b; c; z)
  where phi is the hypergeometric function.

  Args: 
    a, b, c, z: input parameters to phi()
    depth: approximating depth

  Returns:
    The approximated value
  """
  k = np.zeros(depth)
  for i in range(depth): 
    if i%2==0:
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
  # print(k)
  # print(f)
  return 1 / f[0]

def RatioHGMGF(y_d, y_x, p, m, depth=1000):
  """
  Compute the ratio of hypergeometric moment-generating functions
    E((q/p)^(2x)) / E((q/p)^(2y))
  where x ~ HG(m, y_d, y_x), y ~ HG(m, y_d + 1, y_x) 
  Note that the increment is on the second parameter (i.e., y_d + 1).

  Args: 
    y_d: observed # of 1s
    p: bit flipping probability
    m: length of each bloom filter
    type: one of "exact" or "asymp"

  Returns:
    A real number of the simulated epsilon
  """
  q = 1 - p
  if y_d + y_x > m: 
    y_d = m - y_d
    y_x = m - y_x
  
  a = - y_d
  b = - y_x
  c = m - y_x - y_d 
  z = (q / p) ** 2
  l = GaussContFrac(a, b, c, z, depth=depth)

  return (m - y_x) / np.maximum(m - y_x - y_d, 1) * l

def evaluateEpsGivenYd(y_d, p, m, type="exact", y_x=None, depth=1000):
  """
  Evaluate the privacy parameter (epsilon) of one BF.

  Args: 
    y_d: observed # of 1s
    p: bit flipping probability
    m: length of each bloom filter
    type: one of "exact" or "asymp"

  Returns:
    A real number of the simulated epsilon
  """
  ## process the inputs
  q = 1 - p
  if y_x == None:
    y_x = np.random.binomial(y_d, q) + np.random.binomial(m - y_d, p)
    if y_d + y_x > m: 
      y_d = m - y_d
      y_x = m - y_x

  if (type == "asymp"): ## normal approximation
    b = 2 * np.log(q / p)
    l = -b * y_d / m + b ** 2 / 2 * y_d * (m - y_d) * (2 * y_x - m) / m ** 3
    l = np.exp(l)
  elif (type == "exact"): ## exact form
    l = RatioHGMGF(y_d=y_d, y_x=y_x, p=p, m=m, depth=depth) 
  
  ## Pr(A(D')=X) / Pr(A(D)=X)
  # R = (l * y_x * q / p + (m - y_x) * p / q) / (l * y_x + m - y_x) ## version 1
  R = p / q / l ## version 2

  return np.log(R)

def findYxGivenYdIs1(p, m, d):
  '''
  Find the 1-d quantile of y_x given y_d = 1.
  '''
  q = 1 - p
  res = np.zeros(2)
  ## pmf of y_x
  v = stats.binom(m - 1, p)
  ## find d/2 quantile
  lower = 0
  upper = m
  while lower + 1 < upper: 
    y_x = (lower + upper) // 2
    y_x_cdf = p * v.cdf(y_x) + q * v.cdf(y_x - 1) 
    if y_x_cdf > d / 2:
      upper = y_x
    else: 
      lower = y_x
  res[0] = y_x

  ## find 1-d/2 quantile
  lower = 0
  upper = m
  while lower + 1 < upper: 
    y_x = (lower + upper) // 2
    y_x_cdf = p * v.cdf(y_x) + q * v.cdf(y_x - 1) 
    if y_x_cdf > 1 - d / 2:
      upper = y_x
    else: 
      lower = y_x
  res[1] = y_x
  
  return res

def findPGivenYdIs1(e, m, d=0.001):
  """
  Find needed flipping probability (p) given y_d = 1. 
  This uses binary search.

  Args: 
    e: privacy parameter (epsilon)
    m: length of each bloom filter

  Returns:
    A real number of the simulated p
  """
  
  lower = 0
  upper = .5
  while lower + 1e-7 < upper:
    p = (lower + upper) / 2
    y_x = findYxGivenYdIs1(p, m, d)
    epsilon = np.array((evaluateEpsGivenYd(1, p, m, depth=1000, y_x=y_x[0]), 
               evaluateEpsGivenYd(1, p, m, depth=1000, y_x=y_x[1])))
    epsilon_hat = np.max(abs(epsilon))
    if epsilon_hat > e:
      lower = p
    else: 
      upper = p

  return p

def estimateE(k, m, p, d=0.001, n_simu=10000):
  """
  Calculate the privacy parameter (epsilon). 

  Args: 
    k: number of publishers
    m: length of each bloom filter
    p: flipping probability
    d: approximation parameter 
    n_simu: number of sampling times; shoule be larger than 1/d

  Returns:
    An estimated of flipping probability p
  """
  q = 1 - p
  P0 = [stats.binom.pmf(y, k, p) for y in range(k+1)]
  Pk = [stats.binom.pmf(y, k, q) for y in range(k+1)]

  ## g|D', take 1-delta quantile of ln(R)
  g0 = stats.multinomial.rvs(m-1, P0, size=n_simu)
  gk = stats.multinomial.rvs(1, Pk, size=n_simu)
  g = g0 + gk
  w = [(q / p) ** (2 * y - k) for y in range(k+1)]
  R = g.dot(w) / m
  e0 = np.quantile(np.log(R), 1 - d)

  ## g|D, take delta quantile of ln(R)
  g = stats.multinomial.rvs(m, P0, size=n_simu)
  w = [(q / p) ** (2 * y - k) for y in range(k+1)]
  R = g.dot(w) / m
  e1 = np.quantile(-np.log(R), 1 - d)

  return max(e0, e1)

def estimateP(k, m, e=np.log(3), d=0.0001, n_simu=100000, tol=1e-5):
  """
  Search for needed flipping probability (p). 

  Args: 
    k: number of publishers
    m: length of each bloom filter
    e: target privacy parameter (epsilon)
    d: approximation parameter 
    n_simu: will be passed to estimateE() funciton. 

  Returns:
    An estimated of flipping probability p
  """
  if tol <= 0:
    raise ValueError('"tol" should be positive.')
  lower = 0
  upper = 0.5
  while lower + tol < upper:
    p = (lower + upper) / 2
    e_hat = estimateE(k, m, p, d=d, n_simu=n_simu)
    if e_hat < e:
      upper = p
    else: 
      lower = p
  return p