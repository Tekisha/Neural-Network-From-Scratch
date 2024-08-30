import numpy as np


class OptimizerAdam:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """
        Initialize the Adam optimizer with the specified parameters.

        Parameters:
        learning_rate (float): Initial learning rate for the optimizer.
        decay (float): Learning rate decay factor.
        epsilon (float): Small value to avoid division by zero in updates.
        beta_1 (float): Decay rate for the first moment (momentum) estimates.
        beta_2 (float): Decay rate for the second moment (cache) estimates.
        """
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        """
        Adjust the current learning rate based on the decay factor before updating parameters.
        """
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        """
        Update the parameters of the layer using Adam optimization.

        Parameters:
        layer (object): The layer whose parameters will be updated.
        """
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1. - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1. - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / (1. - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1. - self.beta_1 ** (self.iterations + 1))

        layer.weight_cache = self.beta_2 * layer.weight_cache + (1. - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1. - self.beta_2) * layer.dbiases ** 2

        weight_cache_corrected = layer.weight_cache / (1. - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1. - self.beta_2 ** (self.iterations + 1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (
                    np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (
                    np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        """
        Increment the iteration count after parameter updates.
        """
        self.iterations += 1
