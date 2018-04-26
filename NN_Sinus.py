import numpy as np
import pickle
from matplotlib import pyplot as plt
from NN import NeuralNet
import sklearn
from sklearn import datasets

x = np.linspace(0, 2*np.pi, 1000)
y = np.sin(x) + np.random.normal(0,0.1,x.shape)

# plt.plot(x,y)
# plt.show()

nn = NeuralNet(3, [1,10,10,1])

nn.stochastic_grad_descent(x,y)

x_predict = np.linspace(0, 2*np.pi, 103)
y_predict = []
for x in x_predict:
    y_predict.append(nn.predict(x))

plt.plot(x_predict,y_predict)
plt.show()