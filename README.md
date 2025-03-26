# Micrograd & MLP from Scratch

This repository contains an educational Jupyter Notebook that implements a minimal automatic differentiation engine (inspired by Andrej Karpathy's micrograd) and builds a simple multilayer perceptron (MLP) from scratch using Python. The project demonstrates the fundamentals of computation graphs, operator overloading, and backpropagation in an intuitive and modular manner.

## Overview

- **Micrograd Engine:**  
  A minimal autograd system implemented via the `Value` class. This class encapsulates a scalar value along with its gradient and builds a dynamic computation graph through operator overloading.

- **Neural Network Components:**  
  - **Neuron:** Implements a single neuron with randomly initialized weights and bias, using a tanh activation function.  
  - **Layer:** Groups multiple neurons to form a neural layer.  
  - **MLP (Multilayer Perceptron):** Stacks several layers to build a fully connected neural network for end-to-end training.

- **Training Example:**  
  A simple training loop is provided where the MLP is trained on a small dataset. The loss is computed as the sum of squared errors, and gradients are updated using gradient descent.

## Features

- **Educational Focus:**  
  Designed to help you understand how automatic differentiation and backpropagation work under the hood.

- **Custom Autograd Implementation:**  
  Demonstrates how operator overloading in Python can be used to build computation graphs and perform gradient propagation.

- **Modular Design:**  
  The project is organized into reusable components (`Value`, `Neuron`, `Layer`, and `MLP`), making it easy to extend and experiment with.

