

# python NeuralNetworkTests.py TestOptimizers1
# python .\NeuralNetworkTests.py TestOptimizers2
class Sgd:
    
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        # print('shapes')
        # print(weight_tensor.shape)
        # print(gradient_tensor.shape)
        return weight_tensor - self.learning_rate * gradient_tensor


class SgdWithMomentum:
    def __init__(self):
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        pass


class Adam:
    def __init__(self):
        pass

    def calculate_update(self, weight_tensor, gradient_tensor):
        pass