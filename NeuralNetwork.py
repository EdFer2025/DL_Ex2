import numpy as np
import copy


# python NeuralNetworkTests.py TestNeuralNetwork1
class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor, self.label_tensor = None, None

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def forward_network(self):
        """
        This method forwards the input through the network
        :return:
        net_output: The output of the network
        net_loss: The loss of the network
        """
        self.input_tensor, self.label_tensor = self.data_layer.next()

        net_output = self.input_tensor
        for layer in self.layers:
            net_output = layer.forward(net_output)

        net_loss = self.loss_layer.forward(net_output, self.label_tensor)
        self.loss.append(net_loss)

        return net_output, net_loss

    def forward(self):
        _, net_loss = self.forward_network()
        return net_loss

    def backward(self):
        layer_loss = self.loss_layer.backward(self.label_tensor)
        for layer in reversed(self.layers):
            layer_loss = layer.backward(layer_loss)
        return

    def train(self, iterations):
        self.loss = []
        print("Training {} iterations".format(iterations))
        for it in range(iterations):
            print_bar(it + 1, iterations, 50)

            self.forward()
            self.backward()
        return

    def test(self, input_tensor):
        net_output = input_tensor
        for layer in self.layers:
            net_output = layer.forward(net_output)
        return net_output


def print_bar(part, total, size):
    from math import ceil
    ratio = part / total
    percent = ceil(ratio * 100)
    past = ceil(ratio * size)
    print("=" * past + "." * (size - past), "{}%".format(percent), "  {}/{}".format(part, total), end="\r")