import numpy as np


class ActivationReLU:
    """
    Rectified Linear Activation (ReLU) class.
    Applies the ReLU activation function element-wise.
    """
    def __init__(self):
        self.output = None

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
