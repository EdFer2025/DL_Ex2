from Layers.Base import BaseLayer
import numpy as np


# python NeuralNetworkTests.py TestSoftMax
class SoftMax(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.weights = None
        self.last_activation = None

    def forward(self, input_tensor):
        input_size = input_tensor.shape
        input_copy = input_tensor.copy()

        # normalizing input
        # x_i' = x_i - max(x)
        input_maxs = np.max(input_copy, axis=1)
        input_maxs = np.repeat(
            input_maxs,
            repeats=input_size[1]
        ).reshape(input_size)
        input_copy = input_copy - input_maxs

        # y' = exp(X)
        result = np.exp(input_copy)

        # y' = exp(x_k) / SUM(exp(x_j)
        sum_vector = np.sum(result, axis=1)
        sum_vector = np.repeat(
            sum_vector,
            repeats=input_size[1]
        ).reshape(input_size)
        result = result / sum_vector

        self.last_activation = result
        return result

    def backward(self, error_tensor):
        error_copy = error_tensor.copy()

        # K = E - SUM(E * y')
        k = np.sum(error_tensor * self.last_activation, axis=1)
        sum_vector = np.repeat(
            k,
            repeats=error_copy.shape[1]
        ).reshape(error_copy.shape)
        error_copy = error_copy - sum_vector

        # y' * K
        result = self.last_activation * error_copy
        return result
