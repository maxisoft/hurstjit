# hurstjit
## Hurst exponent evaluation and R/S-analysis

![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![Build Status](https://travis-ci.org/maxisoft/hurstjit.svg?branch=main)](https://travis-ci.org/maxisoft/hurstjit)

**hurstjit** is a small Python module for analysing __random walks__ and evaluating the __Hurst exponent (H)__.

H = 0.5 — Brownian motion,  
0.5 < H < 1.0 — persistent behavior,  
0 < H < 0.5 — anti-persistent behavior.  

## Installation
Install **hurstjit** module with
`pip install git+https://github.com/maxisoft/hurstjit`

## Note
This is a fork from [Mottl/hurst](https://github.com/Mottl/hurst) which use numba jit in order to get great performance boost.  
Exemple code is more than 100x time faster (timeit from 1.67 sec to 12.2 ms)


## Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from hurstjit import compute_Hc, random_walk

# Use random_walk() function or generate a random walk series manually:
# series = random_walk(99999, cumprod=True)
np.random.seed(42)
random_changes = 1. + np.random.randn(99999) / 1000.
series = np.cumprod(random_changes)  # create a random walk from random changes

# Evaluate Hurst equation
H, c, data = compute_Hc(series, kind='price', simplified=True)

# Plot
f, ax = plt.subplots()
ax.plot(data[0], c * data[0] ** H, color="deepskyblue")
ax.scatter(data[0], data[1], color="purple")
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Time interval')
ax.set_ylabel('R/S ratio')
ax.grid(True)
plt.show()

print("H={:.4f}, c={:.4f}".format(H, c))
```

![R/S analysis](https://github.com/Mottl/hurst/raw/master/examples/regression.png?raw=true "R/S analysis")

```H=0.4964, c=1.4877```

### Kinds of series
The `kind` parameter of the `compute_Hc` function can have the following values:  
`'change'`: a series is just random values (i.e. `np.random.randn(...)`)  
`'random_walk'`: a series is a cumulative sum of changes (i.e. `np.cumsum(np.random.randn(...))`)  
`'price'`: a series is a cumulative product of changes (i.e. `np.cumprod(1+epsilon*np.random.randn(...)`)

## Brownian motion, persistent and antipersistent random walks
You can generate random walks with `random_walk()` function as following:

### Brownian
```brownian = random_walk(99999, proba=0.5)```


![Brownian motion](https://github.com/Mottl/hurst/raw/master/examples/Brownian_motion.png?raw=true "Brownian motion")

### Persistent
```persistent = random_walk(99999, proba=0.7)```


![Persistent random walk](https://github.com/Mottl/hurst/raw/master/examples/Persistent.png?raw=true "Persistent random walk")

### Antipersistent
```antipersistent = random_walk(99999, proba=0.3)```


![Antipersistent random walk](https://github.com/Mottl/hurst/raw/master/examples/Antipersistent.png?raw=true "Antipersistent random walk")
