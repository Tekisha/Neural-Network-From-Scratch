import numpy as np
from activation_softmax import ActivationSoftmax
from loss_categorical_cross_entropy import LossCategoricalCrossEntropy


class ActivationSoftmaxLossCategoricalcrossentropy:
    """Softmax classifier - combined Softmax activation
     and cross-entropy loss for faster backward step"""
    def __init__(self):
        """Initialize the softmax activation and cross-entropy loss."""
        self.dinputs = None
        self.output = None
        self.activation = ActivationSoftmax()
        self.loss = LossCategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        """
        Perform the forward pass by applying the softmax activation
        on the inputs and then calculating the loss using cross-entropy.

        :param inputs: The input values (logits).
        :param y_true: The true labels.
        :return: The calculated loss.
        """
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        """
        Perform the backward pass by calculating the gradient of the
        loss with respect to the inputs.

        :param dvalues: The gradient values from the next layer.
        :param y_true: The true labels.
        """
        samples = len(dvalues)
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples

    def regularization_loss(self, layer):
        """
        Calculate the regularization loss from the layer's weights and biases.

        :param layer: The layer to calculate regularization loss from.
        :return: The total regularization loss.
        """
        regularization_loss = 0
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss
