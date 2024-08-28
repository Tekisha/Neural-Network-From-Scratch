import numpy as np


class ActivationReLU:
    """
    Rectified Linear Activation (ReLU) class.
    Applies the ReLU activation function element-wise.
    """

    def __init__(self):
        self.dinputs = None
        self.inputs = None
        self.output = None

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0
