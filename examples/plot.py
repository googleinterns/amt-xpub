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


"""Create plots for privacy evaluation of "shuffling+flipping"."""


from buffling import snr
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

## global parameters
NUM_PUBS = 4
FLIP_PROB = 0.05
# plt.style.use("ggplot")

## dependent parameters
NUM_LOGO = 2 ** NUM_PUBS
TRANSIT_MAT = snr.generate_transition_matrix(NUM_PUBS, FLIP_PROB)


## heatmap: SNR vs (i, j)
fig, ax = plt.subplots(1,4,figsize=(16, 3))
for i in range(4): 
  skewness = 3 * i 
  input_logo_wgt = np.exp(- skewness * (np.arange(NUM_LOGO) + 1) / NUM_LOGO)
  input_logo_wgt /= sum(input_logo_wgt)
  s = snr.calculate_snr(input_logo_wgt, TRANSIT_MAT)
  sns.heatmap(np.sqrt(s), ax = ax[i])
  ax[i].set_title(f"SNR (skewness: {skewness})")
plt.savefig('docs/fig/1_snr_vs_ij.pdf')
plt.close()


## line plot: maximizing coefficients vs skewness
data = pd.DataFrame(columns=['skewness', 'logo', 'coefficient'])
data = data.astype({'logo': 'category'})
for skewness in (np.arange(6) * 2):
  input_logo_wgt = np.exp(- skewness * (np.arange(NUM_LOGO) + 1) / NUM_LOGO)
  input_logo_wgt /= sum(input_logo_wgt)
  s,c = snr.calculate_snr_ij(0, NUM_LOGO-1, input_logo_wgt, TRANSIT_MAT, maximizer=True)
  for i in range(len(c)):
    data.loc[len(data)] = [skewness, i, c[i]]
plt.figure(figsize=(6,4))
sns.scatterplot(x="logo", y="coefficient", hue='skewness', data=data)
sns.lineplot(x="logo", y="coefficient", hue='skewness', legend=False, data=data)
plt.hlines(0, -.5, NUM_LOGO-1, colors="grey", linestyles="dashed")
plt.title(f"logo-counts coefficients of SNR maximizer ({NUM_PUBS} pubs)")
plt.savefig('docs/fig/2_max_logo_coefs.pdf')
plt.close()


## line plot: SNR vs skewness
skewness_breaks = np.arange(8 * NUM_PUBS ** 2, step=NUM_PUBS ** 2 / 2)
data = pd.DataFrame(columns=['skewness', 'SNR'])
for skewness in skewness_breaks:
  input_logo_wgt = np.exp(- skewness * (np.arange(NUM_LOGO) + 1) / NUM_LOGO)
  input_logo_wgt /= sum(input_logo_wgt)
  s = snr.calculate_worst_snr(np.array(input_logo_wgt), TRANSIT_MAT)
  data.loc[len(data)] = [skewness, s]
plt.figure(figsize=(6,4))
sns.scatterplot(x="skewness", y="SNR", data=data)
sns.lineplot(x="skewness", y="SNR", data=data)
plt.title(f"signal-to-noise ratio ({NUM_PUBS} pubs)")
plt.savefig('docs/fig/3_snr_vs_skewness.pdf')
plt.close()
