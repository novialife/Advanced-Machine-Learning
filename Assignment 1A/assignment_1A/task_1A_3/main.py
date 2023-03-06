from UGMM import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# Generate data
COMP = 3
SAMPLE  = 10000

mu_arr = np.random.choice(np.arange(-10, 10, 2), COMP) + np.random.random(COMP)
X = [np.random.normal(loc=mu, scale=1, size=SAMPLE) for mu in mu_arr]

models = [UGMM(X[_]) for _ in range(COMP)]

# Fit the model
for i in range(COMP):
    models[i].fit(max_iter=1000)

fig, ax = plt.subplots(figsize=(15, 4))
for i in range(len(models)):
    sns.distplot(X[i], ax=ax, hist=True, norm_hist=True)
    sns.distplot(np.random.normal(models[i].mu, 1, SAMPLE), color='k', hist=False, kde=True)

plt.show()
fig.savefig('UGMM.png')
