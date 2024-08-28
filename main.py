import numpy as np
from nnfs.datasets import spiral_data

from activation_relu import ActivationReLU
from activation_softmax_loss_categorical_cross_entropy import ActivationSoftmaxLossCategoricalcrossentropy
from layer_dense import LayerDense
from optimizer_sgd import OptimizerSGD


def main():
    # Create dataset
    X, y = spiral_data(samples=100, classes=3)
    # Create Dense layer with 2 input features and 64 output values
    dense1 = LayerDense(2, 64)
    # Create ReLU activation (to be used with Dense layer):
    activation1 = ActivationReLU()
    # Create second Dense layer with 64 input features (as we take output
    # of previous layer here) and 3 output values (output values)
    dense2 = LayerDense(64, 3)
    # Create Softmax classifier’s combined loss and activation
    loss_activation = ActivationSoftmaxLossCategoricalcrossentropy()
    # Create optimizer
    optimizer = OptimizerSGD()

    for epoch in range(10001):
        # Perform a forward pass of our training data through this layer
        dense1.forward(X)
        # Perform a forward pass through activation function
        # takes the output of first dense layer here
        activation1.forward(dense1.output)
        # Perform a forward pass through second Dense layer
        # takes outputs of activation function of first layer as inputs
        dense2.forward(activation1.output)
        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        loss = loss_activation.forward(dense2.output, y)

        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch {epoch}, loss {loss:.3f}, accuracy {accuracy:.3f}')
        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        optimizer.update_params(dense1)
        optimizer.update_params(dense2)


if __name__ == "__main__":
    main()
