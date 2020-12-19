import numpy as np


class NeuralNetwork:
    def __init__(self, gateInput, gateOutput, lr):
        np.random.seed(1)
        self.lr = lr

        self.gateInput = gateInput
        self.gateOutput = gateOutput
        self.input_shape = (1, np.shape(gateInput[0])[0])  # ugly!!
        self.output_shape = (1, np.shape(gateOutput[0])[0])  # ugly!!
        self.layer_1_nodes = 5
        self.layer_2_nodes = 5
        self.layer_3_nodes = 5

        self.weights_1 = 2 * np.random.random((self.input_shape[1], self.layer_1_nodes)) - 1
        self.weights_2 = 2 * np.random.random((self.layer_1_nodes, self.layer_2_nodes)) - 1
        self.weights_3 = 2 * np.random.random((self.layer_2_nodes, self.layer_3_nodes)) - 1
        self.out_weights = 2 * np.random.random((self.layer_3_nodes, self.output_shape[1])) - 1

    @staticmethod
    def relu(x):
        return [np.max(_x, 0) for _x in x]

    @staticmethod
    def relu_d(x):
        return [np.heaviside(x, 1) for _x in x]

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    @staticmethod
    def classify(x):
        x[x < 0.5] = 0
        x[x >= 0.5] = 1
        return x

    def think(self, x):
        # Multiply the input with weights and find its sigmoid activation for all layers
        layer1 = self.sigmoid(np.dot(x, self.weights_1))
        layer2 = self.sigmoid(np.dot(layer1, self.weights_2))
        layer3 = self.sigmoid(np.dot(layer2, self.weights_3))
        output = self.classify(np.dot(layer3, self.out_weights))
        print(layer1)
        print(layer2)
        print(layer3)
        return output

    def train(self, num_steps):
        for x in range(num_steps):
            # Same as code of thinking
            layer1 = self.sigmoid(np.dot(self.gateInput, self.weights_1))
            layer2 = self.sigmoid(np.dot(layer1, self.weights_2))
            layer3 = self.sigmoid(np.dot(layer2, self.weights_3))
            output = self.sigmoid(np.dot(layer3, self.out_weights))

            outputError = self.gateOutput - output

            delta = outputError * self.sigmoid_derivative(output) * self.lr

            out_weights_adjustment = np.dot(layer3.T, delta)

            self.out_weights += out_weights_adjustment

            delta = np.dot(delta, self.out_weights.T) * self.sigmoid_derivative(layer3)
            weight_3_adjustment = np.dot(layer2.T, delta)
            self.weights_3 += weight_3_adjustment

            delta = np.dot(delta, self.weights_3.T) * self.sigmoid_derivative(layer2)
            weight_2_adjustment = np.dot(layer1.T, delta)
            self.weights_2 += weight_2_adjustment

            delta = np.dot(delta, self.weights_2.T) * self.sigmoid_derivative(layer1)
            weight_1_adjustment = np.dot(self.gateInput.T, delta)
            self.weights_1 += weight_1_adjustment
