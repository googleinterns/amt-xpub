from buffling import evaluation
import numpy as np
import pandas as pd

data = pd.DataFrame(columns=['m', 'p_hat', 'type'])
data = data.astype({'m': 'int', 'type': 'category'})
for m in (1000, 2000, 5000, 10000, 20000, 50000, 
          100000, 200000, 500000, 1000000, 2000000):
  p_hat = evaluation.findPGivenYdIs1(e=np.log(3), m=m, d=0.0001)
  data.loc[len(data)] = [m, p_hat, "empirical"]

print(data.pivot(index='m', columns='type', values='p_hat'))

p = evaluation.estimateP(10, 1000000, d=0.0001, n_simu=500000) 
print(p)