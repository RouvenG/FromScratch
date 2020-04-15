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


    c1 = np.random.multivariate_normal(mean_1,cov_1, number_of_samples)
    c2 = np.random.multivariate_normal(mean_2,cov_2, number_of_samples)


    # c1 = c1[np.linalg.norm(c1,axis=1) > 2]
    # c2 = c2[np.linalg.norm(c2,axis=1) < 2]
    print(c2)

    return c1,c2

def create_binary_test_data2(number_of_samples):


    points = np.random.uniform(-10,10,(number_of_samples,2))
    dec = lambda x: 0.2*x**3 + x**2

    c1 = []
    c2 = []
    for x in points:
        if x[1] - dec(x[0]) > 1:
            c1.append(x)
        elif x[1] - dec(x[0]) < -1:
            c2.append(x)

    c1 = np.array(c1)
    c2 = np.array(c2)

    return c1, c2

c1,c2 = create_binary_test_data2(1000)
print(c1)
print(c2)
#testdata = open('x3.pickle',"wb")
#pickle.dump((c1, c2),testdata)


plt.scatter(c1[:,0],c1[:,1])
plt.scatter(c2[:,0],c2[:,1])
plt.show()