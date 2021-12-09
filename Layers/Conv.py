from numpy.random.mtrand import weibull
from Layers.Base import BaseLayer
import numpy as np 
from scipy.signal import correlate, convolve

# python .\NeuralNetworkTests.py TestConv
# python .\SoftConvTests.py
class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True
        self.weights_optimizer = None
        self.bias_optimizer = None

        self.stride_shape = stride_shape # can be a single value or a tuple
        self.convolution_shape = convolution_shape #  1D or a 2D convolution layer        
        self.input_channels = convolution_shape[0]
        self.num_kernels = num_kernels
        self.weights_shape = [num_kernels] + list(self.convolution_shape)

        self.batch_size = None
        # Initialize the parameters of this layer uniformly random in the range [0; 1).
        
        self.weights = np.random.random(size=self.weights_shape)
        self.bias = np.random.random(size=(num_kernels,))

        self.input_dimensions = None # dimensions of the input not including the channels
        self.input_shape = None
        self.input_tensor = None
        self.output_shape = None # (self.batch_size, self.num_kernels, self.input_dimensions...)

        self.gradient_weights = None
        self.gradient_bias = None

    """def set_gradient_weights(self, gradient_weights):
        self.gradient_weights = gradient_weights

    def get_gradient_weights(self):
        return self.gradient_weights
    
    gradient_weights = property(get_gradient_weights, set_gradient_weights)
    """
    """def set_gradient_bias(self, gradient_bias):
        self.gradient_bias = gradient_bias
    
    def get_gradient_bias(self):
        return self.gradient_bias
    
    gradient_bias = property(get_gradient_bias, set_gradient_bias)
    """

    def initialize(self, weights_initializer, bias_initializer):
        # TODO understand what means fan_in, and fan_out
        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])
        self.weights = weights_initializer.initialize(self.weights_shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.num_kernels, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        self.input_tensor = input_tensor
        
        self.batch_size = self.input_shape[0]
        self.input_dimensions = self.input_shape[2:]

        batch_output_shape = [self.batch_size, self.num_kernels] + list(self.input_dimensions)
        batch_output = np.zeros(shape=batch_output_shape)

        for e in range(self.batch_size):
            sample = input_tensor[e]
            for i in range(self.weights.shape[0]):
                w = self.weights[i]
                out = correlate(sample, w, mode="same", method="direct")   
                out = np.sum(out, axis=0)
                out = out + self.bias[i]
                batch_output[e][i] = out  

        # sampling using strides
        self._check_stride_shape()
        
        idxs = []
        for i in range(len(self.input_dimensions)):
            ax = self.input_dimensions[i]
            idx = list(range(0, ax, self.stride_shape[i]))
            idxs.append(idx)
        # print(idxs)

        sampled_batch_output = None

        if len(self.input_dimensions) == 1: # 1D
            sampled_batch_output = np.zeros(shape=(self.batch_size, self.num_kernels, len(idxs[0])))
            for b in range(self.batch_size):
                for k in range(self.num_kernels):
                    sampled_batch_output[b, k, :] = batch_output[b, k, idxs[0]]
            return sampled_batch_output
        elif len(self.input_dimensions) == 2: # 2D
            sampled_batch_output = np.zeros(shape=(self.batch_size, self.num_kernels, len(idxs[0]), len(idxs[1])))
            for b in range(self.batch_size):
                for k in range(self.num_kernels):
                    for i in range(len(idxs[0])):
                        sampled_batch_output[b, k, i, :] = batch_output[b, k, idxs[0][i], idxs[1]]
            return sampled_batch_output
        return batch_output

    def _check_stride_shape(self):
        if type(self.stride_shape) == 'int':
            self.stride_shape = [self.stride_shape] * self.input_dimensions
        elif type(self.stride_shape) == 'tuple':
            self.stride_shape = list(self.stride_shape)
        
        # chech there is a stride for all dimensions
        if len(self.stride_shape) < len(self.input_dimensions):
            self.stride_shape = self.stride_shape + [1]*(len(self.input_dimensions) - len(self.stride_shape))


    def backward(self, error_tensor):
        backward_kernels_shape = [self.input_channels, self.num_kernels] + list(self.convolution_shape[1:])
        backward_kernels = np.zeros(shape=backward_kernels_shape)
       
        for k in range(self.num_kernels):
            for c in range(self.input_channels):
                kern = np.flip(self.weights[k,c], axis=0)
                # if len(self.convolution_shape) > 2: # 2D convolution
                #     kern = np.flip(kern, axis=2)
                # kern = new_weights[k,c]
                backward_kernels[c, k] = kern
        

        # Compute input_gradient
        input_gradient = np.zeros(shape=self.input_shape)
        for b in range(self.batch_size):
            for k in range(backward_kernels.shape[0]):
                out = convolve(error_tensor[b], backward_kernels[k], mode="same", method="direct")
                out = np.sum(out, axis=0)                
                input_gradient[b, k] = out
        
        # compute weights_gradient
        for b in range(self.batch_size):
            in_tensor = self.input_tensor[b]
            for e in range(error_tensor.shape[0]):
                err_tensor = error_tensor[e]
                out = convolve(in_tensor, err_tensor, mode="same", method="direct")
                


            pass
        return input_gradient