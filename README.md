# Neural Network from Scratch

This project implements a neural network from scratch using Python and NumPy. It includes essential components like layers, activation functions, loss functions, and optimizers, enabling you to build and train a neural network without relying on high-level libraries like TensorFlow or PyTorch.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Components](#components)
  - [Layers](#layers)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Optimizers](#optimizers)

## Features

- Custom Layer Implementations: Dense, Dropout layers with support for regularization.
- Activation Functions: Softmax activation for classification tasks.
- Loss Functions: Categorical Cross-Entropy for multi-class classification.
- Optimizers: SGD, Adagrad, RMSprop, and Adam optimizers.
- Training and Evaluation: Framework for building, training, and evaluating neural networks.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/Tekisha/Neural-Network-From-Scratch
cd Neural-Network-From-Scratch
pip install numpy 
```

## Usage

To use the neural network framework, follow these steps:

1. **Create Layers**: Define your network architecture by creating and stacking different layers.

2. **Set Activations and Losses**: Choose the activation functions and loss functions appropriate for your task.

3. **Select an Optimizer**: Choose one of the implemented optimizers to update your network's parameters.

4. **Train the Model**: Feed data to the network, compute the loss, and update the weights using the optimizer.

5. **Evaluate the Model**: Test the network's performance on unseen data.

## Components

### Layers

- **LayerDense**: 
  - A fully connected layer with weights and biases. This layer performs a linear transformation on the input data.
  - **Constructor Parameters**:
    - `n_inputs`: Number of inputs to the layer.
    - `n_neurons`: Number of neurons in the layer.
    - `weight_regularizer_l1`: L1 regularization strength for weights.
    - `weight_regularizer_l2`: L2 regularization strength for weights.
    - `bias_regularizer_l1`: L1 regularization strength for biases.
    - `bias_regularizer_l2`: L2 regularization strength for biases.

- **LayerDropout**: 
  - A dropout layer used for regularization to prevent overfitting by randomly setting a fraction of input units to zero during training.
  - **Constructor Parameters**:
    - `rate`: Fraction of input units to drop during training.

### Activation Functions

- **ActivationReLU**:
  - The ReLU (Rectified Linear Unit) activation function, which introduces non-linearity by setting all negative values to zero.
  - **Usage**: Typically used after dense layers to introduce non-linearity.

- **ActivationSoftmax**:
  - The Softmax activation function used for multi-class classification tasks. It converts logits to probabilities by exponentiating and normalizing the outputs.
  - **Usage**: Typically used as the final activation function in classification problems.

### Loss Functions

- **LossCategoricalCrossEntropy**:
  - Computes the categorical cross-entropy loss for classification tasks. This loss measures the difference between the true labels and the predicted probabilities.
  - **Methods**:
    - `forward(y_pred, y_true)`: Computes the loss given predictions and true labels.
    - `calculate(output, y)`: Calculates the mean loss over all samples.
    - `backward(dvalues, y_true)`: Computes the gradient of the loss with respect to the predictions.

### Optimizers

- **OptimizerSGD**:
  - Stochastic Gradient Descent optimizer with optional momentum and learning rate decay.
  - **Constructor Parameters**:
    - `learning_rate`: Learning rate for parameter updates.
    - `decay`: Learning rate decay over iterations.
    - `momentum`: Momentum factor for updates.

- **OptimizerAdagrad**:
  - Adagrad optimizer which adapts the learning rate based on the historical gradient.
  - **Constructor Parameters**:
    - `learning_rate`: Learning rate for parameter updates.
    - `decay`: Learning rate decay over iterations.
    - `epsilon`: Small constant to avoid division by zero.

- **OptimizerRMSprop**:
  - RMSprop optimizer which adjusts the learning rate based on a moving average of squared gradients.
  - **Constructor Parameters**:
    - `learning_rate`: Learning rate for parameter updates.
    - `decay`: Learning rate decay over iterations.
    - `epsilon`: Small constant to avoid division by zero.
    - `rho`: Decay factor for the moving average of squared gradients.

- **OptimizerAdam**:
  - Adam optimizer which combines the advantages of both RMSprop and momentum-based optimization.
  - **Constructor Parameters**:
    - `learning_rate`: Learning rate for parameter updates.
    - `decay`: Learning rate decay over iterations.
    - `epsilon`: Small constant to avoid division by zero.
    - `beta_1`: Exponential decay rate for the first moment estimates.
    - `beta_2`: Exponential decay rate for the second moment estimates.



