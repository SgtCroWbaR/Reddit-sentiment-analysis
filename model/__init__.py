import numpy as np
from math import sqrt, log2
from tqdm import tqdm

np.random.seed(69)


def cross_entropy(p, q):
    return -sum([p[i] * log2(q[i]) for i in range(len(p))])


def relu(x):
    return max(0.0, x)


def none(x):
    pass


class Dense:
    def __init__(self, input_size, output_size, activation=relu):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(output_size, input_size) * sqrt(2 / input_size)
        self.bias = np.zeros((output_size, 1))
        self.activation = np.vectorize(activation)

    def forward(self, input_vec):
        self.input_vec = input_vec
        self.output_vec = self.activation(np.dot(self.weights, self.input_vec) + self.bias)
        return self.output_vec

    def backward(self, grad):
        self.delta_weights = np.dot(grad, self.input_vec) / self.input_size
        self.delta_bias = np.sum(grad, axis=1, keepdims=True) / self.input_size
        next_grad = np.dot(self.weights.T, grad)
        next_grad = grad * self.activation(next_grad)
        return next_grad

    def update(self, learning_rate):
        self.weights -= learning_rate * self.delta_weights
        self.bias -= learning_rate * self.delta_bias


class Sequential:
    def __init__(self, layers=None, loss_function=cross_entropy):
        self.loss_function = np.vectorize(loss_function)
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

    def train(self, x, y, epochs, batch_size, learning_rate):
        self.history = {}
        for epoch in tqdm(range(epochs), desc="Epochs"):
            for i in range(0, x.shape[0], batch_size):
                x_batch = x[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                output = self.forward(x_batch)
                loss = self.loss_function(output, y_batch)
                grad = (output - y_batch) / x_batch.shape[0]
                self.bacward(grad)
                self.update(learning_rate)
            self.history[epoch] = loss
            print(f"Loss: {loss}")

    def predict(self, x):
        return self.forward(x)
