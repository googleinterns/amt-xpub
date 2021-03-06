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


"""Evaluate signal-to-noise ratio of the 'shuffling+flipping' algorithm."""

import numpy as np
import scipy as sp
import pandas as pd


def generate_transition_matrix(n_pub, p=0.1, first2=False, tol=1e-15):
  '''Generate the transition matrix of logos. 

  Args: 
    n_pub: Integer, number of publishers.
    p: Flipping probablity.
    first2: Boolean, whether return only the first 2 rows.
    tol: A small Real number. This will be add to the sum of each row, 
      followed by a row normalizing. 
  
  Return: 
    A numpy.array of n_pub x n_pub, containing the transition probability. 
  '''
  ## check input
  assert int(n_pub) == n_pub and n_pub > 0, "n_pub must be a postive integer." 
  assert p > 0 and p < 1, "p can only range between (0,1)"

  n_logo = 2 ** n_pub
  n_row = 2 if first2 else n_logo
  transition_mat = np.zeros((n_row, n_logo))
  ## use binary representation and compute hamming distance
  for i in range(n_row):
    for j in range(n_logo):
      n_p = bin(i ^ j)[2:].count('1')
      transition_mat[i,j] = p**n_p * (1 - p)**(n_pub - n_p)
  rs = transition_mat.sum(axis=1) + tol ## add a small number to row sums
  return transition_mat / rs[:, np.newaxis]


def calculate_signal_to_noise_ratio_of_logo_counts(
  i, j, input_logo_wgt, transition_mat, maximizer=False):
  '''Calculate the SNR with fixed incremental change of two logo counts.

  This is a helper function to calculate_snr(). The two logo counts changed 
  are indexed by i and j. 

  Args: 
    i, j: Integer, indices of the two logo-counts that are changed. 
    input_logo_wgt: Input logo-counts weights. 
    transition_mat: Transition matrix of logo-counts with blipping. 
    maximizer: Boolean, whether return the linear coefficients.

  Return: 
    A real number of SNR if maximizer is False. Otherwise, 
    both the SNR and the maximizing logo-counts coefficients. 
  '''
  ## check input 
  assert i <= transition_mat.shape[0] and j <= transition_mat.shape[1], \
    "Illegal i or j."

  ## compute the B matrix
  D = np.diag(input_logo_wgt.dot(transition_mat))
  F = np.diag(input_logo_wgt)
  B = D - (transition_mat.transpose()).dot(F.dot(transition_mat))

  ## incremental query from logo j to logo i
  a = transition_mat[j,:] - transition_mat[i,:]
  ## solve for one set of linear coefficients
  c = sp.linalg.solve(B, a)
  ## normalize to unite length
  c -= np.mean(c) 
  c /= np.linalg.norm(c)

  ## compute the maximizing SNR
  snr = sum((c * a) ** 2) / np.dot(c.transpose(), B.dot(c))

  if maximizer: 
    return snr, c
  return snr


def calculate_signal_to_noise_ratio(input_logo_wgt, transition_mat):
  """Calculate the signal-to-noise ratio for all possible incremental change.

  Args: 
    input_logo_wgt: Input logo-counts weights. 
    transition_mat: Transition matrix of logo-counts with blipping. 

  Return: 
    A real number, the maximal SNR.
  """
  n_logo = transition_mat.shape[0]

  ## brute-force all possible incremental change from logo i to logo j
  s = np.zeros(transition_mat.shape)
  for i in range(n_logo):
    for j in range(n_logo):
      if i == j:
        continue
      s[i,j] = calculate_signal_to_noise_ratio_of_logo_counts(
        i, j, input_logo_wgt, transition_mat) 
  return s


def calculate_max_signal_to_noise_ratio(input_logo_wgt, transition_mat):
  """Calculate the max signal-to-noise ratio. 

  Calculate the signal-to-noise ratio upon the incremental change 
  from logo i to logo j.

  Args: 
    input_logo_wgt: Input logo-counts weights. 
    transition_mat: Transition matrix of logo-counts with blipping. 

  Return: 
    A real number, the maximal signal-to-noise ratio.
  """
  n_logo = transition_mat.shape[0]
  i = 0
  j = n_logo - 1
  return calculate_signal_to_noise_ratio_of_logo_counts(
    i, j, input_logo_wgt, transition_mat) 
