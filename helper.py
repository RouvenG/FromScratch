"""
2/24/18
@author: Rouven Glauert
@email: rouvenglauert@gmail.com
@license: BSD 

Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt

def logistic(x):
    return 1.0 / (1.0 + np.exp(-x))

def deriv_logistic(x):
    return logistic(x)*(1-logistic(x))

# x = np.arange(-100, 100, 1)
#
# plt.plot(x, logistic(x))
# plt.plot(x, deriv_logistic(x))
# plt.show()