import numpy as np
from nnfs.datasets import spiral_data

from activation_relu import ActivationReLU
from activation_softmax_loss_categorical_cross_entropy import ActivationSoftmaxLossCategoricalcrossentropy
from layer_dense import LayerDense
from layer_dropout import LayerDropout
from optimizer_adagrad import OptimizerAdagrad
from optimizer_adam import OptimizerAdam
from optimizer_rmsprop import OptimizerRMSprop
from optimizer_sgd import OptimizerSGD


def main():
    # Create dataset
    X, y = spiral_data(samples=1000, classes=3)

    dense1 = LayerDense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
    activation1 = ActivationReLU()
    dropout1 = LayerDropout(0.1)
    dense2 = LayerDense(64, 3)

    # Create Softmax classifier's combined loss and activation
    loss_activation = ActivationSoftmaxLossCategoricalcrossentropy()

    # Create optimizer
    #optimizer = OptimizerSGD(decay=8e-8, momentum=0.9)
    #optimizer = OptimizerAdagrad(decay=1e-4)
    #optimizer = OptimizerRMSprop(decay=1e-4)
    #optimizer = OptimizerRMSprop(learning_rate=0.02, decay=1e-5, rho=0.999)
    optimizer = OptimizerAdam(learning_rate=0.02, decay=1e-5)

    for epoch in range(10001):
        dense1.forward(X)
        activation1.forward(dense1.output)
        dropout1.forward(activation1.output)
        dense2.forward(dropout1.output)

        # Perform a forward pass through the activation/loss function
        # takes the output of second dense layer here and returns loss
        data_loss = loss_activation.forward(dense2.output, y)
        regularization_loss = loss_activation.regularization_loss(dense1) + loss_activation.regularization_loss(dense2)
        loss = data_loss + regularization_loss

        # Calculate accuracy from output of activation2 and targets
        # calculate values along first axis
        predictions = np.argmax(loss_activation.output, axis=1)
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        accuracy = np.mean(predictions == y)

        if not epoch % 100:
            print(f'epoch: {epoch}, ' +
                  f'acc: {accuracy:.3f}, ' +
                  f'loss: {loss:.3f}, ' +
                  f'lr: {optimizer.current_learning_rate}')

        # Backward pass
        loss_activation.backward(loss_activation.output, y)
        dense2.backward(loss_activation.dinputs)
        dropout1.backward(dense2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)

        # Update weights and biases
        optimizer.pre_update_params()
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.post_update_params()

    #testing model
    X_test, y_test = spiral_data(samples=100, classes=3)
    dense1.forward(X_test)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y_test)
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y_test.shape) == 2:
        y_test = np.argmax(y_test, axis=1)
    accuracy = np.mean(predictions == y_test)
    print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')


if __name__ == "__main__":
    main()
