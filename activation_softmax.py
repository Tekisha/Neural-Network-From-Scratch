import numpy as np


class ActivationSoftmax:
    """
    Softmax activation class.
    Applies the Softmax activation function to each row of the input data.
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)
