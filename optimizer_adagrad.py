import numpy as np


class OptimizerAdagrad:
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        """
        Initialize the Adagrad optimizer with the specified parameters.

        Parameters:
        learning_rate (float): Initial learning rate for the optimizer.
        decay (float): Learning rate decay factor.
        epsilon (float): Small value to avoid division by zero in updates.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.epsilon = epsilon
        self.iterations = 0

    def pre_update_params(self):
        """
        Apply decay to the learning rate before updating the parameters.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of the layer using Adagrad optimization.

        Parameters:
        layer (object): The layer whose parameters will be updated.
        """
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        """
        Increment the iteration count after parameter updates.
        """
        self.iterations += 1
