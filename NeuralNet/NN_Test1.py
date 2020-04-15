import numpy as np
import pickle
from matplotlib import pyplot as plt
from NN import NeuralNet
import sklearn
from sklearn import datasets

X, t = sklearn.datasets.make_moons(2000, noise=0.10)
plt.scatter(X[:,0], X[:,1], s=40, c=t, cmap=plt.cm.Spectral)
plt.show()

nn = NeuralNet(3, [1,3,3,1])

# Test

def plot_dec_bound():
    x = np.linspace(-1.5,1.5,100)
    y = np.linspace(-1.5,1.5,100)
    l = []
    for i in x:
        for j in y:
            l.append([i,j])

    c1 = []
    c2 = []
    for p in l:
        cla = nn.predict(p)
        #print(cla)
        cla = np.argmax(cla)

        if cla == 0:
            c1.append(p)
        if cla == 1:
            c2.append(p)



    c1 = np.array(c1)
    c2 = np.array(c2)
    plt.scatter(c1[:,0],c1[:,1],s=1)
    plt.scatter(c2[:,0],c2[:,1],s=1)
    # plt.scatter(x1[:100,0],x1[:100,1])
    # plt.scatter(x2[:100,0],x2[:100,1])
    plt.show()

while True:
    c = nn.stochastic_grad_descent(X, t)
    try:
        plot_dec_bound()
    except:
        print("###No Boundary###")
        pass


c = np.array(c)

plt.plot(c[:,0])
plt.plot(c[:,1])
plt.plot(c[:,0] + c[:,1])
plt.show()
