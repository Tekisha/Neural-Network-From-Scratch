import numpy as np


class OptimizerRMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        """
        Initialize the RMSprop optimizer with the specified parameters.

        Parameters:
        learning_rate (float): Initial learning rate for the optimizer.
        decay (float): Learning rate decay factor.
        epsilon (float): Small value to avoid division by zero in updates.
        rho (float): Decay rate for the moving average of squared gradients.
        """
        self.learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.rho = rho
        self.iterations = 0
        self.current_learning_rate = learning_rate

    def pre_update_params(self):
        """
        Adjust the current learning rate based on the decay factor before updating parameters.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of the layer using RMSprop optimization.

        Parameters:
        layer (object): The layer whose parameters will be updated.
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Increment the iteration count after parameter updates.
        """
        self.iterations += 1

