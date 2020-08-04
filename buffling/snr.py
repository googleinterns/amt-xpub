import numpy as np
import scipy as sp
import pandas as pd

# @title helpers
def generateTransitionMatrix(k, p=0.1, first2=False, tol=1e-15):
  '''
  Generate the transition matrix of logos. 

  Args: 
    k: number of publishers
    p: flipping probablity
    first2: whether return only the first 2 rows
  '''
  u = 2**k
  r = 2 if first2 else u
  P = np.zeros((r, u))
  for i in range(r):
    for j in range(u):
      n_p = bin(i ^ j)[2:].count('1')
      P[i,j] = p**n_p * (1 - p)**(k - n_p)
  rs = P.sum(axis=1) + tol ## add a small number to row sums
  return P / rs[:, np.newaxis]

def simulateMargin(f, P, index=1):
  g = 0
  for i in range(len(f)):
    g += sp.stats.binom.rvs(f[i], P[i,index])
  return g

def calculateSNRij(i,j,f,P,maximizer=False):
  '''
  Helper function to calculateSNR(f, P).

  Args: 
    maximizer: return the linear coefficients.
  '''
  D = np.diag(f.dot(P))
  F = np.diag(f)
  B = D - (P.transpose()).dot(F.dot(P))
  a = P[i,:] - P[j,:]
  c = sp.linalg.solve(B, a)
  c -= np.mean(c) 
  c /= np.linalg.norm(c)
  snr = sum((c * a) ** 2) / np.dot(c.transpose(), B.dot(c))
  if maximizer: 
    return snr, c
  return snr

def calculateSNR(f, P):
  u = P.shape[0]
  s = np.zeros(P.shape)
  for i in range(u):
    for j in range(u):
      if i == j:
        continue
      s[i,j] = calculateSNRij(i,j,f,P) 
  return s

def calculateWorstSNR(f, P):
  u = P.shape[0]
  i = 0
  j = u - 1
  return calculateSNRij(i,j,f,P) 
