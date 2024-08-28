import numpy as np
from activation_softmax import ActivationSoftmax
from loss_categorical_cross_entropy import LossCategoricalCrossEntropy


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class ActivationSoftmaxLossCategoricalcrossentropy:
    def __init__(self):
        self.dinputs = None
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)
        # Set the output
        self.output = self.activation.output
        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples
