import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from activation_relu import ActivationReLU
from activation_softmax import ActivationSoftmax
from loss_categorical_cross_entropy import LossCategoricalCrossEntropy

nnfs.init()


class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the LayerDense with weights and biases.

        Parameters:
        n_inputs (int): Number of inputs to the layer.
        n_neurons (int): Number of neurons in the layer.
        """
        self.output = None
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Perform the forward pass for the layer.

        Parameters:
        inputs (np.ndarray): Input data to the layer.
        """
        self.output = np.dot(inputs, self.weights) + self.biases


# Example usage:
if __name__ == "__main__":
    X, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLU()
    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftmax()
    loss_function = LossCategoricalCrossEntropy()
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y)
    print("Loss:" + str(loss))

    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)
    print("Accuracy:" + str(accuracy))
