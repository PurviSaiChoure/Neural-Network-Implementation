# XOR Neural Network with Backpropagation Demonstration

This repository contains a simple neural network implementation designed to solve the XOR problem. The network architecture is pre-configured with specific weights and biases to demonstrate correct XOR computation without training. The code also illustrates the backpropagation process for educational purposes.

## Overview
The XOR problem is a classic challenge in neural networks, requiring a model to learn the non-linear relationship between inputs and outputs. This implementation uses a feedforward neural network with one hidden layer to achieve the correct XOR outputs.

### Network Architecture
- **Input Layer**: 2 neurons (for XOR inputs)
- **Hidden Layer**: 2 neurons (with sigmoid activation)
- **Output Layer**: 1 neuron (with sigmoid activation)

Pre-trained weights and biases are hardcoded to:
- Input-to-Hidden Weights: `[[20, 20], [-20, -20]]`
- Hidden Biases: `[-10, 30]`
- Hidden-to-Output Weights: `[[20], [20]]`
- Output Bias: `[-30]`

## Implementation Details
- **Sigmoid Activation**: Used in both hidden and output layers.
- **Forward Propagation**: Computes predictions through the network.
- **Backward Propagation**: Demonstrates gradient calculations (weights are not updated in this example).

## Usage
1. **Dependencies**: Requires `numpy`.
2. **Run the Code**:
   ```python
   python FNN.py 
