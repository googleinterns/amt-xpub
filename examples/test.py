from buffling import privacy
import numpy as np
import pandas as pd


# global parameters
EPSILON = np.log(3)
DELTA = 1e-4
NUM_PUB = 8
NUM_BIT = 1000000


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
print(f"\n{NUM_PUB} publishers, {NUM_BIT} bits, epsilon = {EPSILON}, delta = {DELTA}")
p = privacy.estimate_p(NUM_PUB, NUM_BIT, e=EPSILON, d=DELTA, n_simu=500000) 
print(f"Required flipping probability = {p}.")