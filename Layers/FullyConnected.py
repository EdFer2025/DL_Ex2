from Layers.Base import BaseLayer
import numpy as np 


# python NeuralNetworkTests.py TestFullyConnected1
class FullyConnected(BaseLayer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()

        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        # weights and bias per each neuron in the layer
        self.weights = np.random.random(size=(input_size + 1, output_size))

        self._optimizer = None
        self.last_input = None
        self._gradient_weights = None
        self._gradient_input = None
    
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gw):
        self._gradient_weights = gw

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def forward(self, input_tensor):
        """
        input tensor is a matrix with input_size columns
        and batch_size rows.
        each input includes a input value 1 for the bias
        """
        batch_size = input_tensor.shape[0]

        bias_ones = np.ones((batch_size,)).reshape((batch_size, 1))
        input_tensor_extended = np.concatenate([input_tensor, bias_ones], axis=1)

        result = np.dot(input_tensor_extended, self.weights)

        self.last_input = input_tensor_extended
        return result
        
    def backward(self, error_tensor):
        """
        returns error vector with shape (batch_size, input_size)
        """

        self._gradient_input = np.dot(error_tensor, self.weights.T )
        result = self._gradient_input[:, :-1]

        self._gradient_weights = np.dot(self.last_input.T, error_tensor)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)

        return result
