"""
2/24/18
@author: Rouven Glauert
@email: rouvenglauert@gmail.com
@license: BSD 

Please feel free to use and modify this, but keep the above information. Thanks!
"""
import helper
import numpy as np
import pickle
from matplotlib import pyplot as plt

class NeuralNet():

    def __init__(self, depth, layersize):
        self.depth = depth
        self.weight_matrices, self.bias_vectors = self.create_layers(depth, layersize)
        self.a = []
        self.z = []

    # h = g(W*x+b)
    def calc_next_layer(self, W, x, b):

        z = np.dot(W, x) + b

        a = self.activation(z)
        # print(W.shape)
        # print(x.shape)
        # print(b.shape)
        # print(z.shape)
        # print(a.shape)
        # print("____________")
        self.z.append(z)
        self.a.append(a)
        return a

    def activation(self, z):
        return helper.logistic(z)

    def d_activation(self, z):
        return helper.deriv_logistic(z)

    def predict(self, x):

        layer = self.calc_next_layer(self.weight_matrices[0], x, self.bias_vectors[0])

        for i in np.arange(1, self.depth):
            output = self.calc_next_layer(self.weight_matrices[i], layer, self.bias_vectors[i])
            layer = output

        return output



    def backprop(self, x, t):
        """
        returns the derivative with respect to all weights
        """
        y = self.predict(x)


        #delta_l[0] is the output layer and so on
        delta_l = [-(t-y)*self.d_activation(self.z[-1])]


        for l in reversed(range(self.depth)):
        #     print(l)
        #     print("weights(transposed):" + str(self.weight_matrices[l].T.shape))
        #     print("delta_l: " + str(delta_l[-1].shape))
        #     print("d_activation: " + str(self.d_activation(self.z[l]).shape))
        #     print("z: " + str(self.z[l].shape))
        #     print("___________")

            delta_l.append(np.dot(self.weight_matrices[l].T, delta_l[-1])*self.d_activation(self.z[l-1]))

        d_w = []
        d_b = []

        for i in range(self.depth):
            d_w.append(np.outer(delta_l[i], self.a[-i]))
            d_b.append(delta_l[i])

        return [d_w, d_b]

    def single_J(self,x,t):
        return np.linalg.norm(self.predict(x)-t)**2/2



    def create_layers(self, depth, layersize):
        """
        :param depth: number of layers of the neural net
        :param layersize: list of integers: number of neurons in each layer
                        the first number is the size of the input layer
        :return: tuple(list of weight matrices, list of bias vectors)
        """
        weight_matrices = []
        bias_vectors = []

        for i in np.arange(1, depth+1):
            weight_matrices.append(np.random.rand(layersize[i],layersize[i-1])/1000)

        for i in np.arange(1, depth+1):
            bias_vectors.append(np.random.rand(layersize[i])/1000)


        return(weight_matrices, bias_vectors)


    def stochastic_grad_descent(self,x):

        l_rate = 0.05


        for d in range(1000):


            # Initialize Gradient for batch
            grad = self.backprop(x[0, 0:2], x[0, 2])

            batch_size = 10



            for b in range(batch_size):
                i = np.random.randint(0, len(x))
                grad_i = self.backprop(x[i, 0:2], x[i, 2])

                #Average over batch
                for dw_l in range(len(grad_i[0])):
                    grad[0][dw_l] += grad_i[0][dw_l]

                for db_l in range(len(grad_i[1])):
                    grad[1][dw_l] += grad_i[1][dw_l]



            d_w = list(reversed(grad[0]))
            d_b = list(reversed(grad[1]))

            print(str(self.single_J([0,8],1)) + "  " + str(self.single_J([12,2],0)))


            for i in range(self.depth):
                self.weight_matrices[i] -= l_rate * d_w[i]/batch_size
                self.bias_vectors[i] -= l_rate * d_b[i]/batch_size





x1,x2 = pickle.load(open('data.pickle',"rb"))

t = np.array([0]*100000 + [1] *100000)
x = np.concatenate((x1,x2))

t = t[:,np.newaxis]
train = np.hstack((x,t))


plt.scatter(x1[:,0],x1[:,1])
plt.scatter(x2[:,0],x2[:,1])
plt.show()

nn = NeuralNet(2, [2, 10, 1])


print(nn.predict(np.array([12,2])))
print(nn.predict(np.array([0,8])))

nn.stochastic_grad_descent(train)

print(nn.predict(np.array([12,2])))
print(nn.predict(np.array([0,8])))
