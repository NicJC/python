import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import uniform
from scipy.stats import gamma
from scipy.stats import hypergeom
from scipy import stats
from scipy.integrate import quad
import seaborn as sns
from functools import partial
from scipy.integrate import quad
import scipy.integrate as integrate
import scipy.special as special
from scipy.stats import expon
from scipy.stats import t
import matplotlib.pyplot as plt
import pandas as pd

rv = norm()
dir(rv)

npoints = 20   # number of integer support points of the distribution minus 1
npointsh = npoints // 2
npointsf = float(npoints)
nbound = 4   # bounds for the truncated normal
normbound = (1+1/npointsf) * nbound   # actual bounds of truncated normal
grid = np.arange(-npointsh, npointsh+2, 1)   # integer grid
gridlimitsnorm = (grid-0.5) / npointsh * nbound   # bin limits for the truncnorm
gridlimits = grid - 0.5   # used later in the analysis
grid = grid[:-1]
probs = np.diff(stats.truncnorm.cdf(gridlimitsnorm, -normbound, normbound))
gridint = grid

normdiscrete = stats.rv_discrete(values=(gridint,
             np.round(probs, decimals=7)), name='normdiscrete')

print('mean = %6.4f, variance = %6.4f, skew = %6.4f, kurtosis = %6.4f' %
      normdiscrete.stats(moments='mvsk'))


nd_std = np.sqrt(normdiscrete.stats(moments='v'))

n_sample = 500
np.random.seed(87655678)   # fix the seed for replicability
rvs = normdiscrete.rvs(size=n_sample)
f, l = np.histogram(rvs, bins=gridlimits)
sfreq = np.vstack([gridint, f, probs*n_sample]).T
print(sfreq)

plt.plot(sfreq)
plt.show()

x1 = np.array([-7, -5, 1, 4, 5], dtype=np.float)
kde1 = stats.gaussian_kde(x1)
kde2 = stats.gaussian_kde(x1, bw_method='silverman')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(x1, np.zeros(x1.shape), 'b+', ms=20)  # rug plot
x_eval = np.linspace(-10, 10, num=200)
ax.plot(x_eval, kde1(x_eval), 'k-', label="Scott's Rule")
ax.plot(x_eval, kde2(x_eval), 'r-', label="Silverman's Rule")

plt.show()
