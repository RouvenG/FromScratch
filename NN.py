"""
2/24/18
@author: Rouven Glauert
@email: rouvenglauert@gmail.com
@license: BSD 

Please feel free to use and modify this, but keep the above information. Thanks!

The implementation is oriented at
http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm
"""
import helper
import numpy as np
import pickle
from matplotlib import pyplot as plt
import random


class NeuralNet():

    def __init__(self, depth, layersize):
        self.depth = depth
        self.weight_matrices, self.bias_vectors = self.create_layers(depth, layersize)
        self.a = []
        self.z = []
        self.log_l_rate = -2.3
        self.log_weight_decay = -2.3
        self.gradiendCheckMode = False


    # h = g(W*x+b)

    def set_GCM(self,status):
        self.gradiendCheckMode = status

    def calc_next_layer(self, W, x, b):
        if isinstance(x, float):
            x = np.array([x])


        # print("\n W: " + str(W) +
        #       "\n x: " + str(x) +
        #       "\n b: " + str(b.shape) +
        #       "\n np.dot(W, x) = " + str(np.dot(W, x).shape)
        #       )

        dotWx = np.dot(W, x)
        # print("x: " + str(x))
        # print("dotWx" + str(dotWx))
        # print("b" + str(b))
        z = dotWx + b.reshape(dotWx.shape)

        a = self.activation(z)
        # print("z: " + str(z))
        # print("a: " + str(a))

        self.z.append(z)
        self.a.append(a)
        return a

    def activation(self, z):
        return helper.reLu(z)

    def d_activation(self, z):
        return helper.deriv_reLu(z)

    def predict(self, x,  **kwargs):
        """
        The feed forward of the a given input
        :param x: Input variable
        :param kwargs: Keyword arguments
        :return: Output of the Neural Net
        """


        if kwargs:
            if kwargs['weight_matrices']:
                weight_matrices = kwargs['weight_matrices']
            if kwargs['bias_vectors']:
                bias_vectors = kwargs['bias_vectors']
        else:
            weight_matrices = self.weight_matrices
            bias_vectors = self.bias_vectors


        self.a.append(np.array(x))

        layer = self.calc_next_layer(weight_matrices[0], x, bias_vectors[0])
        for i in np.arange(1, self.depth):
            output = self.calc_next_layer(weight_matrices[i], layer, bias_vectors[i])
            layer = output



        return output

    def backprop(self, x, t):
        """
        returns the derivative with respect to all weights
        """
        y = self.predict(x)
        #print(x)
        #print(t)

        #delta_l[0] is the output layer and so on
        # print(self.z)
        # print(self.a)
        delta_l = [-(t-y)*self.d_activation(self.z[-1])]
        # print(delta_l)

        for l in reversed(range(self.depth)):
            # print(l)
            # print("weights(transposed):" + str(self.weight_matrices[l].T.shape))
            # print("delta_l: " + str(delta_l[-1].shape))
            # print("d_activation: " + str(self.d_activation(self.z[l]).shape))
            # print("z: " + str(self.z[l].shape))
            # print("___________")

            delta_l.append(np.dot(self.weight_matrices[l].T, delta_l[-1])*self.d_activation(self.z[l-1]))

        d_w = []
        d_b = []

        a = list(reversed(self.a))

        for l in range(self.depth):
            d_b.append(delta_l[l])


        delta_l = delta_l[:-1]
        a = a[1:]

        for l in range(self.depth):
            # print("a[l]: " + str(a[l].T))
            # print("delta_l[l]: " + str(delta_l[l]))
            # print("np.outer(delta_l[l], a[l].T) " + str(np.outer(delta_l[l], a[l].T)))
            d_w.append(np.outer(delta_l[l], a[l])/2.)


        self.a = []
        self.z = []

        if self.gradiendCheckMode:
            d_w_num, d_b_num = self.numerical_grad_check(x,t)
            d_w_num = list(reversed(d_w_num))
            d_b_num = list(reversed(d_b_num))
            # for w_i in zip(d_w_num,d_w):
            #     print(np.abs(w_i[0]-w_i[1]) < 0.00001)
            print("####Numeric#####")
            print(d_w_num)
            print("####Exact#######")
            print(d_w)


        return [d_w, d_b] # dw first entry is last layer

    def batch_backprop(self, X, T, w_decay):
        """

        :param X: The training data batch of samples
        :param T: The training data batch targets
        :return: gradient of the batch
        """

        batch_size = X.shape[0]

        grad = self.backprop(X[0], T[0])

        for b in range(batch_size): # loop over samples
            grad_i = self.backprop(X[b], T[b])

            # Average over batch
            for dw_l in range(len(grad_i[0])): # loop over weight layers dw
                grad[0][dw_l] += grad_i[0][dw_l]

            for db_l in range(len(grad_i[1])): # loop over bias layers db
                grad[1][dw_l] += grad_i[1][dw_l]

        weights = list(reversed(self.weight_matrices))



        for dw_l in range(len(grad[0])):# loop over bias and weight layers
            grad[0][dw_l] /= batch_size
            grad[0][dw_l] += w_decay * np.sum(weights[dw_l])
            grad[1][dw_l] /= batch_size

        #print("Batch", grad)
        return grad

    def single_J(self, x, t, **kwargs):
        """
        :param x: A single datapoint
        :param t: The class belonging to the data sample
        :return:
        """
        #TODO the case that both keyword arguments are given is not properly handled
        if kwargs:
            if kwargs['weight_matrices']:
                y = self.predict(x, weight_matrices=kwargs['weight_matrices'], bias_vectors=kwargs['bias_vectors'])
        else:
            y = self.predict(x)

        return np.linalg.norm(y-t)**2/2

    def J(self, X, T, l):
        """
        :param X:
        :param T:
        :param l:
        :return:
        """

        J_1 = 0
        J_2 = 0

        for (x, t) in zip(X, T):
            J_1 += self.single_J(x, t)
        #TODO change len(T) because if the target has more then one dimension this does not fit
        J_1 /= len(T)

        for w in self.weight_matrices:

            w = w**2
            J_2 += np.sum(w)

        J_2 *= l/2
        return [J_1, J_2]

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
            weight_matrices.append(np.random.uniform(-0.2,0.2,(layersize[i],layersize[i-1])))

        for i in np.arange(1, depth+1):
            bias_vectors.append(np.random.uniform(-0.05,0.05,layersize[i]))


        return(weight_matrices, bias_vectors)

    def stochastic_grad_descent(self, X, T):
        """

        :param X: The train data array
        :param T: The target data array
        :return:
       # """

        def print_progress(l_rate, weight_decay, J, i):
            if i % 10 == 0:
                print("J_samples | J_weight |learning_rate |  weight decay| J ")
            print("%0.5f   | %0.5f  | %0.5f         | %0.5f | %0.5f  " % (J[0],J[1],l_rate,weight_decay,J[0]+J[1]))


        cost_function = []
        J = 1
        it = 0
        low = 1
        while J > 0.18 :
            it += 1
            batch_size = 10

            samples = random.sample(range(len(X)), batch_size)
            grad = self.batch_backprop(X[samples],T[samples],np.exp(self.log_weight_decay))
            d_w = list(reversed(grad[0]))
            d_b = list(reversed(grad[1]))

            if(it%100 == 0):
                c = self.J(X,T, np.exp(self.log_weight_decay))
                cost_function.append(c)
                print_progress(np.exp(self.log_l_rate), np.exp(self.log_weight_decay), c, it/50)
                J = c[0] + c[1]

            if(it%100 == 0):
                self.log_l_rate -= 0.01
                self.log_weight_decay -= 0.01

            #print(d_w)
            for i in range(self.depth):
                self.weight_matrices[i] -= np.exp(self.log_l_rate) * d_w[i]
                self.bias_vectors[i] -= np.exp(self.log_l_rate) * d_b[i]

            # if it > 1000:
            #     break

        return cost_function

    def numerical_grad_check(self,x,t,epsilon=0.00001):

        w = self.weight_matrices
        b = self.bias_vectors

        # Create Empty numerical Gradients
        weights_gradient_num = []
        bias_gradient_num = []

        for w_i in self.weight_matrices:
            weights_gradient_num.append(np.zeros(w_i.shape))

        for b_i in self.bias_vectors:
            bias_gradient_num.append(np.zeros(b_i.shape))


        #Calculate numerical Gradien

        for i in range(len(w)):
            for j in range(w[i].shape[0]):
                for k in range(w[i].shape[1]):

                    w[i][j,k] = self.weight_matrices[i][j,k] + epsilon
                    J_plus = self.single_J(x,t, weight_matrices=w, bias_vectors=b)
                    w[i][j, k] = self.weight_matrices[i][j,k] - epsilon
                    J_minus = self.single_J(x,t, weight_matrices=w, bias_vectors=b)

                    w[i][j, k] = self.weight_matrices[i][j,k]

                    grad = (J_plus - J_minus)/(2*epsilon)

                    weights_gradient_num[i][j,k] = grad


        for i in range(len(b)):
            for j in range(b[i].shape[0]):
                    b[i][j] = self.bias_vectors[i][j] + epsilon
                    J_plus = self.single_J(x,t, weight_matrices=w, bias_vectors=b)
                    b[i][j] = self.bias_vectors[i][j] - epsilon
                    J_minus = self.single_J(x,t, weight_matrices=w, bias_vectors=b)

                    b[i][j] = self.bias_vectors[i][j]

                    grad = (J_plus - J_minus)/(2*epsilon)

                    bias_gradient_num[i][j] = grad



        return (weights_gradient_num,bias_gradient_num)



        # w = self.weight_matrices
        # b = self.bias_vectors
        #
        # flatten_weights = np.reshape(self.weight_matrices[0],[1,self.weight_matrices[0].size])
        #
        # for w_i in self.weight_matrices[1:]:
        #     w_i = np.reshape(w_i, [1, w_i.size])
        #     flatten_weights = np.hstack((w_i, flatten_weights))
        #
        # flatten_bias = np.reshape(self.bias_vectors[0], [1, self.bias_vectors[0].size])
        #
        # for b_i in self.bias_vectors[1:]:
        #     b_i = np.reshape(b_i, [1, b_i.size])
        #     flatten_bias = np.hstack((b_i, flatten_bias))
        #
        # flat_weight_grad = []


