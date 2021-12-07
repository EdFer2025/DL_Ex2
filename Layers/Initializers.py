from Layers.Base import BaseLayer
import numpy as np

# python .\SoftConvTests.py TestInitializers
class Constant:
    def __init__(self, cons_init=0.1):
        self.const_init = cons_init

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.full(weights_shape, self.cons_init)
    

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.random(size=weights_shape)
    

class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = (2/(fan_in + fan_out))**(1/2)
        return np.random.normal(
            loc=0, 
            scale=sigma, 
            size=weights_shape)
    

class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = (2/fan_in)**(1/2)
        return np.random.normal(
            loc=0, 
            scale=sigma, 
            size=weights_shape)

    