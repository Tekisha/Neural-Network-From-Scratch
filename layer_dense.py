import numpy as np
import nnfs
from nnfs.datasets import spiral_data

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
    dense1.forward(X)
    print(dense1.output)
