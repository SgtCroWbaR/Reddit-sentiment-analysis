import numpy as np
from math import sqrt, log2
from tqdm import tqdm

np.random.seed(69)


def cross_entropy(p, q):
    epsilon = 1e-10
    return -sum([p[i] * np.log(q[i] + epsilon) for i in range(p.shape[0])])


def relu(x):
    return max(0.0, x)


def dRelu(x):
    if x < 0:
        return 0
    else:
        return 1


def linear(x):
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dSigmoid(x):
    return x*(1-x)


class Dense:
    def __init__(self, input_size, output_size, activation=relu):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))
        if activation == relu:
            self.derivate = dRelu
        elif activation == sigmoid:
            self.derivate = dSigmoid

        self.activation = np.vectorize(activation)
        self.derivate = np.vectorize(self.derivate)

    def forward(self, input_vec):
        self.input_vec = input_vec
        self.output_vec = self.activation(np.dot(self.weights, self.input_vec) + self.bias)
        return self.output_vec

    def backward(self, grad):
        self.delta_weights = np.dot(grad, self.input_vec.T) / self.input_size
        self.delta_bias = np.sum(grad, axis=1, keepdims=True) / self.input_size
        next_grad = np.dot(self.weights.T, grad)
        next_grad = next_grad * np.sum(self.derivate(self.output_vec))
        return next_grad

    def update(self, learning_rate):
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias


class Sequential:
    def __init__(self, layers=None, loss_function=cross_entropy):
        self.loss_function = loss_function
        if layers is None:
            self.layers = []
        else:
            self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def bacward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def train(self, Xs, Ys, epochs, learning_rate):
        self.history = {}
        loss = float("inf")
        epochs = tqdm(range(epochs))
        for epoch in epochs:
            epochs.set_description(f"Loss: {loss}")
            loss_arr = []
            for x, y in list(zip(Xs, Ys)):
                output = self.forward(x)
                loss = self.loss_function(output, y)
                loss_arr.append(loss)
                grad = (output.T - y) / x.shape[0]
                grad = grad.T
                self.bacward(grad)
                self.update(learning_rate)
            self.history[epoch] = np.average(loss_arr)

    def predict(self, x):
        return self.forward(x)
