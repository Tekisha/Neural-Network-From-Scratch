import numpy as np


class LayerDropout:
    def __init__(self, rate):
        """
        Initialize the LayerDropout with the specified dropout rate.

        Parameters:
        rate (float): Dropout rate (fraction of neurons to drop during training).
        """
        self.dinputs = None
        self.output = None
        self.binary_mask = None
        self.inputs = None
        self.rate = 1 - rate

    def forward(self, inputs):
        """
        Perform the forward pass by applying dropout to the input values.

        Parameters:
        inputs (np.ndarray): Input data to the layer.
        """
        self.inputs = inputs
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        """
        Perform the backward pass, applying the binary mask to the gradient.

        Parameters:
        dvalues (np.ndarray): Gradients of the loss with respect to the layer's outputs.
        """
        self.dinputs = dvalues * self.binary_mask
