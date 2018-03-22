"""
3/16/18
@author: Rouven Glauert
@email: rouvenglauert@gmail.com
@license: BSD 

Please feel free to use and modify this, but keep the above information. Thanks!
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

def create_binary_test_data(number_of_samples):


    mean_1 = 10*np.random.rand(2)
    cov_1 = np.array([[1,np.random.rand(1)],[np.random.rand(1),1]])
    cov_2 = np.array([[1,np.random.rand(1)],[np.random.rand(1),1]])
    mean_2 = 10*np.random.rand(2)


    x1 = np.random.multivariate_normal(mean_1,cov_1, number_of_samples)
    x2 = np.random.multivariate_normal(mean_2,cov_2, number_of_samples)

    return x1,x2

x1,x2 = create_binary_test_data(100000)

testdata = open('data.pickle',"wb")
pickle.dump((x1,x2),testdata)


plt.scatter(x1[:,0],x1[:,1])
plt.scatter(x2[:,0],x2[:,1])
plt.show()