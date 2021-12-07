import numpy as np

# python NeuralNetworkTests.py TestCrossEntropyLoss
class CrossEntropyLoss:

    def __init__(self):
        self.last_prediction = None

    def forward(self, prediction_tensor, label_tensor):
        batch_size = label_tensor.shape[0]
        self.last_prediction = prediction_tensor

        eps = np.finfo('float').eps
        k = np.argmax(label_tensor, axis=1)
        k_prediction = np.zeros(shape=(batch_size,))
        for i in range(batch_size):
            k_prediction[i] = prediction_tensor[i, k[i]] + eps

        loss = np.sum(np.log(k_prediction) * -1)
        return loss

    def backward(self, label_tensor):
        result = (label_tensor / self.last_prediction) * -1
        return result