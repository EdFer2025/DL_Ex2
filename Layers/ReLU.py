from Layers.Base import BaseLayer


# python NeuralNetworkTests.py TestReLU
class ReLU(BaseLayer):
    def __init__(self):
        self.trainable = False
        self.weights = None
        self.last_input = None

    def forward(self, input_tensor):
        self.last_input = input_tensor.copy()
        input_tensor[input_tensor <= 0] = 0
        return input_tensor

    def backward(self, error_tensor):
        result = error_tensor.copy()
        result[self.last_input <= 0] = 0
        return result