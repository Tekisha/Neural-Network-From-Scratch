import numpy as np


class OptimizerSGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        """
        Initialize the SGD (Stochastic Gradient Descent) optimizer with the specified parameters.

        Parameters:
        learning_rate (float): Initial learning rate for the optimizer.
        decay (float): Learning rate decay factor.
        momentum (float): Momentum factor for accelerating updates in the relevant direction.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        """
        Adjust the current learning rate based on the decay factor before updating parameters.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of the layer using SGD optimization.

        Parameters:
        layer (object): The layer whose parameters will be updated.
        """
        if self.momentum:
            if not hasattr(self, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        """
        Increment the iteration count after parameter updates.
        """
        self.iterations += 1
