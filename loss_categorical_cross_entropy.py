import numpy as np


class LossCategoricalCrossEntropy:
    def __init__(self):
        """
        Initialize the LossCategoricalCrossEntropy with necessary attributes.
        """
        self.dinputs = None
        self.correct_confidences = None

    def forward(self, y_pred, y_true):
        """
        Perform the forward pass, calculating the negative log likelihood loss.

        Parameters:
        y_pred (np.ndarray): Predicted probabilities for each class.
        y_true (np.ndarray): True class labels, can be sparse or one-hot encoded.

        Returns:
        np.ndarray: Calculated negative log likelihood losses for each sample.
        """
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            self.correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            self.correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(self.correct_confidences)
        return negative_log_likelihoods

    def calculate(self, output, y):
        """
        Calculate the mean loss over all samples.

        Parameters:
        output (np.ndarray): Predicted probabilities.
        y (np.ndarray): True labels.

        Returns:
        float: Mean loss value.
        """
        sample_losses = self.forward(output, y)
        return np.mean(sample_losses)
    
    def backward(self, dvalues, y_true):
        """
        Perform the backward pass, calculating the gradient of the loss
        with respect to the predictions.

        Parameters:
        dvalues (np.ndarray): Gradients of the loss with respect to predictions.
        y_true (np.ndarray): True class labels.
        """
        samples = len(dvalues)
        labels = len(dvalues[0])
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        self.dinputs = -y_true/dvalues
        self.dinputs = self.dinputs/samples
