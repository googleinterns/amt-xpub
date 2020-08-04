from buffling import snr
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

## global parameters
k = 4
p = 0.05
# plt.style.use("ggplot")

## dependent parameters
u = 2 ** k
P = snr.generateTransitionMatrix(k, p)

## line plot: SNR vs skewness
a_breaks = np.arange(8 * k ** 2, step=k ** 2 / 2)
data = pd.DataFrame(columns=['skewness', 'SNR'])
for a in a_breaks:
  f = np.exp(- a * (np.arange(u) + 1) / u)
  f /= sum(f)
  s = snr.calculateWorstSNR(np.array(f), P)
  data.loc[len(data)] = [a, s]
plt.figure(figsize=(6,4))
sns.scatterplot(x="skewness", y="SNR", data=data)
sns.lineplot(x="skewness", y="SNR", data=data)
plt.title(f"signal-to-noise ratio ({k} pubs)")
plt.savefig('1.pdf')
plt.close()

## heatmap: SNR vs (i, j)
fig, ax =plt.subplots(1,4,figsize=(17, 3))
for i in range(4): 
  a = 3 * i 
  f = np.exp(- a * (np.arange(u) + 1) / u)
  f /= sum(f)
  s = snr.calculateSNR(f, P)
  sns.heatmap(np.sqrt(s), ax = ax[i])
  ax[i].set_title(f"SNR (skewness: a = {a})")
plt.savefig('2.pdf')
plt.close()

## line plot: maximizing coefficients vs skewness
data = pd.DataFrame(columns=['skewness', 'logo', 'coefficient'])
data = data.astype({'logo': 'category'})
for a in (np.arange(6) * 2):
  f = np.exp(- a * (np.arange(u) + 1) / u)
  f /= sum(f)
  s,c = snr.calculateSNRij(0, u-1, f, P, maximizer=True)
  for i in range(len(c)):
    data.loc[len(data)] = [a, i, c[i]]
plt.figure(figsize=(6,4))
sns.scatterplot(x="logo", y="coefficient", hue='skewness', data=data)
sns.lineplot(x="logo", y="coefficient", hue='skewness', legend=False, data=data)
plt.hlines(0, -.5, u-1, colors="grey", linestyles="dashed")
plt.title(f"logo-counts coefficients of SNR maximizer ({k} pubs)")
plt.savefig('3.pdf')
plt.close()
