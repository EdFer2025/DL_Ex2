from Layers.Base import BaseLayer
import numpy as np 

# python .\NeuralNetworkTests.py TestFlatten
class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.batch_size = None
        self.input_shape = None

    def forward(self, input_tensor):
        self.batch_size = input_tensor.shape[0]
        self.input_shape = input_tensor.shape[1:]
        return input_tensor.reshape((self.batch_size, np.prod(self.input_shape)))

    def backward(self, error_tensor):        
        s = [self.batch_size] + list(self.input_shape)
        return error_tensor.reshape(s)
