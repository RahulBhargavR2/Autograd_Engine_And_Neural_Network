# Autograd Engine + Neural Network

A lightweight deep-learning framework built **from scratch in Python**, implementing the fundamental mechanisms behind automatic differentiation, tensor computation, backpropagation, neural-network training, and optimization.

The project began as a **scalar reverse-mode automatic differentiation engine** and progressively evolved into a tensor-based engine capable of handling:

* Reverse-mode automatic differentiation
* Dynamic computational graphs
* Gradient accumulation
* Tensor operations using native Python lists
* Broadcasting
* Matrix multiplication
* Gradient propagation
* Linear neural-network layers
* Activation functions
* Loss functions
* Mini-batch training
* SGD, RMSprop, and Adam
* Computational-graph visualization
* Model persistence

The objective is **not to replace PyTorch or TensorFlow**. Instead, the project focuses on understanding and implementing the mechanisms that modern deep-learning frameworks provide behind high-level APIs.

---

## Table of Contents

* [Overview](#overview)
* [Project Goals](#project-goals)
* [Features](#features)
* [Architecture](#architecture)
* [Scalar Autograd Engine](#scalar-autograd-engine)
* [Tensor Engine](#tensor-engine)
* [Tensor Operations](#tensor-operations)
* [Broadcasting](#broadcasting)
* [Gradient Accumulation](#gradient-accumulation)
* [Neural Network Components](#neural-network-components)
* [Activation Functions](#activation-functions)
* [Loss Functions](#loss-functions)
* [Mini-Batch Training](#mini-batch-training)
* [Optimizers](#optimizers)
* [Computational Graph Visualization](#computational-graph-visualization)
* [Model Persistence](#model-persistence)
* [Repository Structure](#repository-structure)
* [Module Responsibilities](#module-responsibilities)
* [Gradient Flow](#gradient-flow)
* [Training Workflow](#training-workflow)
* [Validation and Gradient Checking](#validation-and-gradient-checking)
* [Design Principles](#design-principles)
* [Technology Stack](#technology-stack)
* [Development Evolution](#development-evolution)
* [Limitations](#limitations)
* [Future Improvements](#future-improvements)
* [Learning Outcomes](#learning-outcomes)
* [Comparison With Modern Frameworks](#comparison-with-modern-frameworks)
* [Project Philosophy](#project-philosophy)
* [Status](#status)
* [Author](#author)
* [License](#license)

---

# Overview

Modern deep-learning frameworks hide a large amount of mathematical and computational machinery behind simple APIs such as:

```python
loss.backward()
optimizer.step()
```

Underneath these calls are several important concepts:

* Computational graphs
* Automatic differentiation
* Chain-rule-based gradient computation
* Backpropagation
* Gradient accumulation
* Tensor operations
* Broadcasting
* Matrix multiplication
* Neural-network layers
* Activation functions
* Loss functions
* Parameter updates
* Optimization algorithms

This project implements these concepts incrementally from first principles.

At a high level, the framework follows this pipeline:

```text
Input Data
    │
    ▼
Tensor Operations
    │
    ├── Addition
    ├── Multiplication
    ├── Matrix Multiplication
    ├── Broadcasting
    └── Activation
    │
    ▼
Prediction
    │
    ▼
Loss
    │
    ▼
Backward Pass
    │
    ▼
Gradients
    │
    ▼
Optimizer
    │
    ▼
Updated Parameters
```

The central idea is simple:

> Build the computation graph during the forward pass and use that graph to propagate gradients backward.

---

# Project Goals

The project was developed with the following goals:

1. Understand reverse-mode automatic differentiation.
2. Implement backpropagation without relying on an external autograd engine.
3. Understand how computational graphs are constructed and traversed.
4. Build a tensor abstraction using native Python lists.
5. Implement tensor operations and their corresponding backward passes.
6. Understand broadcasting and gradient reduction.
7. Implement matrix multiplication and its analytical gradients.
8. Build trainable neural-network layers from scratch.
9. Implement common activation and loss functions.
10. Understand mini-batch training internally.
11. Implement optimization algorithms from their mathematical definitions.
12. Visualize computational graphs.
13. Persist trained model parameters.
14. Understand the relationship between mathematical operations and neural-network training.

---

# Features

## Automatic Differentiation

The engine implements reverse-mode automatic differentiation using a dynamically constructed computational graph.

Each differentiable operation records the information required to propagate gradients during the backward pass.

The general process is:

```text
Forward Pass
     │
     ▼
Build Computational Graph
     │
     ▼
Compute Loss
     │
     ▼
Topological Traversal
     │
     ▼
Reverse Traversal
     │
     ▼
Apply Local Derivatives
     │
     ▼
Accumulate Gradients
```

The implementation follows the chain rule:

$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial x}
$$

---

# Architecture

The project has two major computational levels:

```text
┌─────────────────────────────────────────────┐
│           Neural Network Layer              │
│                                             │
│  Linear Layers                              │
│  Activations                                │
│  Loss Functions                             │
│  Training                                   │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│              Tensor Engine                  │
│                                             │
│  Tensor                                     │
│  Element-wise Operations                    │
│  Broadcasting                               │
│  Matrix Multiplication                      │
│  Gradient Propagation                       │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│             Autograd Engine                 │
│                                             │
│  Computational Graph                        │
│  Reverse-mode Autodiff                      │
│  Topological Backward Pass                  │
│  Gradient Accumulation                      │
└─────────────────────────────────────────────┘
```

The scalar implementation provides the conceptual foundation.

The tensor implementation generalizes those ideas to multi-dimensional data represented using **nested native Python lists**.

---

# Scalar Autograd Engine

The original implementation is located in:

```text
scalar/
```

The central abstraction is:

```text
Value
```

A `Value` stores information such as:

```text
data
grad
_prev
_backward
```

The computational graph is constructed dynamically as operations are performed.

For example:

```text
a ─────┐
       │
       ▼
    multiply ───► c
       ▲
       │
b ─────┘
```

The resulting node retains references to the values that contributed to it.

Calling:

```python
loss.backward()
```

performs the essential stages of reverse-mode automatic differentiation:

1. Traverse the graph.
2. Construct a topological ordering.
3. Traverse the graph in reverse.
4. Compute local derivatives.
5. Propagate gradients.
6. Accumulate gradients at shared nodes.

This scalar implementation served as the foundation for the later tensor engine.

---

# Tensor Engine

The tensor implementation is located under:

```text
Tensor/tensor/
```

The central abstraction is:

```text
Tensor
```

Unlike frameworks that rely on NumPy arrays or external tensor libraries, this implementation represents tensor data using **native Python lists**.

For example:

```python
x = [
    [1, 2],
    [3, 4]
]
```

Tensor operations, shape handling, broadcasting, matrix multiplication, and gradient propagation are implemented manually.

The engine supports operations conceptually such as:

```python
a + b
a * b
a @ b
```

while preserving the information required for automatic differentiation.

### Core tensor responsibilities

The tensor implementation handles:

* Tensor data
* Tensor shapes
* Operations
* Parent dependencies
* Gradient storage
* Backward functions
* Gradient accumulation
* Broadcasting
* Gradient reduction
* Matrix multiplication

The important architectural distinction is:

```text
Python Lists
     │
     ▼
Tensor Abstraction
     │
     ▼
Numerical Operations
     │
     ▼
Computational Graph
     │
     ▼
Backward Propagation
```

---

# Tensor Operations

## Element-wise Addition

For:

$$
z=x+y
$$

the local derivatives are:

$$
\frac{\partial z}{\partial x}=1
$$

and:

$$
\frac{\partial z}{\partial y}=1
$$

During backpropagation, the upstream gradient is therefore propagated to both operands.

---

## Element-wise Multiplication

For:

$$
z=xy
$$

the derivatives are:

$$
\frac{\partial z}{\partial x}=y
$$

and:

$$
\frac{\partial z}{\partial y}=x
$$

The backward operation therefore uses the value of the opposite operand when computing each gradient.

---

# Matrix Multiplication

Matrix multiplication is one of the most important operations for neural-network layers.

For:

$$
C=AB
$$

the gradients are:

$$
\frac{\partial L}{\partial A}
=
\frac{\partial L}{\partial C}B^T
$$

and:

$$
\frac{\partial L}{\partial B}
=
A^T\frac{\partial L}{\partial C}
$$

The implementation handles the corresponding shape transformations required by the backward pass.

Matrix multiplication was validated using different matrix configurations, including:

* Square matrices
* Non-square matrices
* Column-vector cases

This validation is important because matrix multiplication gradients are highly dependent on correct tensor dimensions.

---

# Broadcasting

Broadcasting allows operations between tensors with compatible shapes.

For example:

```text
A: (batch, features)

b: (features,)
```

The operation:

```text
A + b
```

can conceptually be treated as:

```text
A
+
[b
 b
 b
 ...]
```

where the same bias vector is applied across the batch dimension.

Because this project does not rely on NumPy for tensor computation, the broadcasting behavior and its backward handling are implemented manually.

## Why broadcasting complicates backpropagation

The forward operation expands a smaller tensor conceptually across one or more dimensions.

However, its gradient must return to the tensor's original shape.

For example:

```text
Forward:

(batch, features)
        +
   (features,)
        │
        ▼
(batch, features)
```

During the backward pass:

```text
(batch, features)
        │
        ▼
Reduce broadcast dimension
        │
        ▼
   (features,)
```

This process is implemented through **gradient unbroadcasting**.

This is particularly important for neural-network biases.

If a bias is shared across an entire batch, every sample contributes to the bias gradient. Those contributions therefore need to be accumulated back into the original bias shape.

---

# Gradient Accumulation

Gradients must be accumulated rather than blindly overwritten.

Consider a computational graph where a tensor contributes to multiple operations:

```text
             ┌── Operation A ──┐
             │                 │
x ───────────┤                 ├──► Loss
             │                 │
             └── Operation B ──┘
```

Both paths contribute to the gradient of `x`.

Therefore:

$$ 
\frac{\partial L}{\partial x} =  \frac{\partial L_A}{\partial x} + \frac{\partial L_B}{\partial x}
$$

The autograd engine accumulates these gradient contributions during the backward pass.

This is a fundamental property of computational graphs.

---

# Neural Network Components

The neural-network functionality is primarily located in:

```text
Tensor/tensor/Linear.py
```

The tensor engine provides the numerical foundation required to construct trainable layers.

A linear layer performs:

$$
y=xW+b
$$

where:

* $x$ is the input
* $W$ is the trainable weight matrix
* $b$ is the trainable bias
* $y$ is the output

The parameters participate in the computational graph.

Therefore:

```text
Input
  │
  ▼
Linear Layer
  │
  ├── Weight
  └── Bias
  │
  ▼
Output
  │
  ▼
Loss
  │
  ▼
Backward
  │
  ├── dL/dW
  └── dL/db
```

This allows the optimizer to update the parameters after gradients have been calculated.

---

# Activation Functions

Activation functions introduce non-linearity into neural networks.

The project includes the core concepts behind commonly used activation functions such as ReLU, Sigmoid, Tanh, and Softmax.

## ReLU

$$
ReLU(x)=\max(0,x)
$$

Its derivative is:

$$
ReLU'(x)=
\begin{cases}
1 & x>0\\
0 & x\leq0
\end{cases}
$$

### Why ReLU introduces non-linearity

Without nonlinear activations, stacking linear transformations still results in a linear transformation.

For example:

$$
W_2(W_1x)
$$

can be represented as another linear transformation.

With an activation function:

$$
W_2\sigma(W_1x)
$$

the network can represent nonlinear relationships.

---

## Sigmoid

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

Its derivative is:

$$
\sigma'(x)=\sigma(x)(1-\sigma(x))
$$

Sigmoid maps values into the range:

$$
(0,1)
$$

---

## Tanh

The hyperbolic tangent is:

$$
\tanh(x)
$$

with derivative:

$$
1-\tanh^2(x)
$$

Its output lies between:

$$
(-1,1)
$$

---

## Softmax

For classification, Softmax converts logits into a probability distribution:

$$
softmax(z_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
$$

The resulting probabilities satisfy:

$$
\sum_i p_i=1
$$

This makes Softmax suitable for converting a vector of class scores into class probabilities.

---

# Loss Functions

The project works with common loss formulations used in neural-network training.

## Mean Squared Error

For regression:

$$
MSE=
\frac{1}{n}
\sum_{i=1}^{n}
(y_i-\hat y_i)^2
$$

MSE measures the average squared difference between the predicted and target values.

---

## Cross-Entropy

For classification, cross-entropy measures the difference between the target distribution and the predicted probability distribution.

When combined with Softmax, it provides a common loss formulation for multi-class classification.

Conceptually:

```text
Logits
  │
  ▼
Softmax
  │
  ▼
Class Probabilities
  │
  ▼
Cross-Entropy
  │
  ▼
Loss
```

---

# Mini-Batch Training

The training process supports mini-batch processing rather than requiring the entire dataset to be processed in a single step.

The general pipeline is:

```text
Dataset
   │
   ▼
Shuffle
   │
   ▼
Create Mini-Batches
   │
   ├── Batch 1
   ├── Batch 2
   ├── Batch 3
   └── ...
   │
   ▼
Forward Pass
   │
   ▼
Loss
   │
   ▼
Backward Pass
   │
   ▼
Optimizer Step
```

If:

```text
batch_size = B
```

and the input contains:

```text
features = d
```

the input batch has the conceptual shape:

$$
X_{batch}\in\mathbb{R}^{B\times d}
$$

The corresponding predictions have a batch dimension as well:

$$
\hat{Y}_{batch}
\in
\mathbb{R}^{B\times output\_dim}
$$

### Shuffling

Before creating batches, the dataset can be shuffled.

Shuffling changes which samples belong to each batch.

It does **not** create separate model parameters for each batch.

The same parameters are reused across all batches:

```text
Batch 1 ──┐
Batch 2 ──┤
Batch 3 ──┼──► Same Model Parameters
Batch 4 ──┤
Batch N ──┘
```

---

# Optimizers

The optimizer implementations are located in:

```text
Tensor/optimizers/
```

Current optimizers include:

```text
optimizer.py
sgd.py
RMSprop.py
Adam.py
```

The optimizer is responsible for converting computed gradients into parameter updates.

This separates two distinct responsibilities:

```text
Autograd
    │
    └── Computes gradients

Optimizer
    │
    └── Uses gradients to update parameters
```

---

# SGD

Stochastic Gradient Descent uses the fundamental update rule:

$$
\theta_{t+1}
=
\theta_t -
\eta\nabla_\theta L
$$

where:

* $\theta$ = model parameter
* $\eta$ = learning rate
* $\nabla_\theta L$ = gradient of the loss

The basic process is:

```text
Parameter
   │
   ▼
Gradient
   │
   ▼
Learning Rate × Gradient
   │
   ▼
Subtract From Parameter
```

SGD provides the fundamental baseline for optimization.

---

# RMSprop

RMSprop maintains an exponentially weighted moving average of squared gradients:

$$
v_t
=
\rho v_{t-1} +
(1-\rho)g_t^2
$$

The parameter update is approximately:

$$
\theta_{t+1}
=
\theta_t -
\frac{\eta}
{\sqrt{v_t+\epsilon}}
g_t
$$

The squared-gradient history provides an adaptive scaling factor for the parameter update.

This allows parameters with consistently large gradients to receive appropriately scaled updates.

---

# Adam

Adam combines momentum-like first-moment tracking with adaptive second-moment tracking.

It maintains:

### First moment

$$
m_t
=
\beta_1m_{t-1} +
(1-\beta_1)g_t
$$

### Second moment

$$
v_t
=
\beta_2v_{t-1} +
(1-\beta_2)g_t^2
$$

Because these estimates are initialized from zero, Adam applies bias correction:

$$
\hat m_t
=
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t
=
\frac{v_t}{1-\beta_2^t}
$$

The final update is:

$$
\theta_{t+1}
=
\theta_t -
\eta
\frac{\hat m_t}
{\sqrt{\hat v_t}+\epsilon}
$$

Conceptually:

```text
Gradient
   │
   ├──────────────► First Moment
   │
   └──────────────► Second Moment
                         │
                         ▼
                  Bias Correction
                         │
                         ▼
                  Adaptive Update
                         │
                         ▼
                    Parameters
```

---

# Optimizer Abstraction

The optimizer implementations follow a common conceptual structure:

```text
Model Parameters
       │
       ▼
Compute Gradients
       │
       ▼
   Optimizer
       │
       ├── SGD
       ├── RMSprop
       └── Adam
       │
       ▼
Update Parameters
```

This separation keeps parameter-update logic independent from the neural-network layer implementation.

The same model can therefore conceptually be trained using different optimization algorithms.

---

# Computational Graph Visualization

The scalar implementation includes graph visualization support through:

```text
scalar/Viizualizer.py
```

Graphviz artifacts are also included:

```text
Digraph.gv
Digraph.gv.svg
```

The visualization makes the computational graph inspectable.

A computation can be represented conceptually as:

```text
       x
      / \
     /   \
    *     +
   / \   / \
  w   b ... ...
      \   /
       \ /
       loss
```

Graph visualization is useful when investigating:

* Incorrect gradients
* Missing dependencies
* Incorrect backward functions
* Gradient accumulation
* Operation ordering
* Computational-graph structure

Instead of treating automatic differentiation as a black box, the graph makes the underlying dependency structure visible.

---

[//]: # ()
[//]: # (# Model Persistence)

[//]: # ()
[//]: # (Model-saving functionality is provided through:)

[//]: # ()
[//]: # (```text)

[//]: # (Tensor/tensor/saveModel.py)

[//]: # (```)

[//]: # ()
[//]: # (The general purpose of model persistence is to separate training from later model usage.)

[//]: # ()
[//]: # (```text)

[//]: # (Training)

[//]: # (   │)

[//]: # (   ▼)

[//]: # (Learned Parameters)

[//]: # (   │)

[//]: # (   ▼)

[//]: # (Save)

[//]: # (   │)

[//]: # (   ▼)

[//]: # (Stored Model)

[//]: # (   │)

[//]: # (   ▼)

[//]: # (Load / Reuse)

[//]: # (```)

[//]: # ()
[//]: # (This avoids requiring the model to be retrained whenever the stored parameters need to be reused.)

[//]: # ()
[//]: # (---)

# Repository Structure

```text
.
├── README.md
│
├── scalar/
│   ├── Digraph.gv
│   ├── Digraph.gv.svg
│   ├── NeuralNetwork.py
│   ├── Value.py
│   └── Viizualizer.py
│
└── Tensor/
    ├── __init__.py
    │
    ├── optimizers/
    │   ├── __init__.py
    │   ├── optimizer.py
    │   ├── sgd.py
    │   ├── RMSprop.py
    │   └── Adam.py
    │
    └── tensor/
        ├── __init__.py
        ├── Linear.py
        └── tensor.py
```

Generated files such as:

```text
__pycache__/
*.pyc
```

are intentionally omitted from the documented source structure.

---

# Module Responsibilities

| Module                           | Responsibility                                             |
| -------------------------------- | ---------------------------------------------------------- |
| `scalar/Value.py`                | Scalar automatic differentiation and computational graph   |
| `scalar/NeuralNetwork.py`        | Neural-network experimentation using scalar autograd       |
| `scalar/Viizualizer.py`          | Computational-graph visualization                          |
| `Tensor/tensor/tensor.py`        | Tensor abstraction and gradient-aware numerical operations |
| `Tensor/tensor/Linear.py`        | Trainable linear neural-network layer                      |
| `Tensor/optimizers/optimizer.py` | Optimizer abstraction                                      |
| `Tensor/optimizers/sgd.py`       | Stochastic Gradient Descent                                |
| `Tensor/optimizers/RMSprop.py`   | RMSprop                                                    |
| `Tensor/optimizers/Adam.py`      | Adam                                                       |

---
[//]: # (| `Tensor/tensor/saveModel.py`     | Model persistence                                          |)
# Gradient Flow

The complete training cycle can be summarized as:

```text
             Input Data
                 │
                 ▼
          ┌──────────────┐
          │ Neural Layer │
          └──────┬───────┘
                 │
                 ▼
            Activation
                 │
                 ▼
              Output
                 │
                 ▼
               Loss
                 │
                 ▼
            backward()
                 │
                 ▼
       Computational Graph
                 │
                 ▼
          Gradients ∂L/∂θ
                 │
                 ▼
             Optimizer
                 │
                 ▼
        Updated Parameters
                 │
                 └──────────────┐
                                │
                                ▼
                         Next Training Step
```

The central separation is:

```text
Forward Pass
    ↓
Compute Loss
    ↓
Backward Pass
    ↓
Compute Gradients
    ↓
Optimizer
    ↓
Update Parameters
```

---

# Training Workflow

A typical training iteration follows this conceptual pattern:

```python
# Forward pass
prediction = model(x)

# Compute loss
loss = loss_function(prediction, target)

# Reset gradients
optimizer.zero_grad()

# Backpropagation
loss.backward()

# Update parameters
optimizer.step()
```

The important architectural distinction is:

> `backward()` computes gradients. It does not determine how model parameters should be updated.

The optimizer is responsible for the parameter update.

Therefore:

```text
backward()
    │
    └── dL/dθ

optimizer.step()
    │
    └── θ ← updated θ
```

This separation is fundamental to the framework design.

---

# Validation and Gradient Checking

A major part of developing an autograd engine is validating the **backward pass independently from the forward pass**.

A forward operation can produce numerically plausible results while still having an incorrect derivative implementation.

Matrix multiplication was explicitly tested with known inputs and analytical gradients.

For example:

```text
A =
[[1, 2],
 [3, 4]]

B =
[[5, 6],
 [7, 8]]
```

For:

$$
C=AB
$$

the implementation's gradient behavior was checked against the analytical derivatives.

Example validated gradients:

```text
dL/dA =
[[11, 15],
 [11, 15]]

dL/dB =
[[4, 4],
 [6, 6]]
```

Gradient validation is particularly important for:

* Matrix multiplication
* Broadcasting
* Shared tensors
* Gradient accumulation
* Neural-network parameters

---

# Design Principles

## 1. Understand the Mathematics

Operations are implemented from their mathematical definitions instead of delegating automatic differentiation to an external framework.

For example, matrix multiplication is not treated as a black-box operation. Its backward equations are explicitly implemented.

---

## 2. Separate Concerns

The project separates:

```text
Autograd
Tensor Computation
Neural-Network Layers
Optimization
Persistence
Visualization
```

Each layer has a distinct responsibility.

---

## 3. Validate Backward Propagation

Forward correctness alone is insufficient for an autograd engine.

The backward implementation must also be validated.

This is particularly important for operations involving tensor dimensions.

---

## 4. Prefer Composability

Operations should be composable:

```text
Tensor
   ↓
Operation
   ↓
Tensor
   ↓
Operation
   ↓
Loss
```

The goal is to allow complete models to be constructed from differentiable primitive operations rather than requiring a manually defined backward pass for every complete model.

---

## 5. Keep Optimization Independent

The model produces gradients.

The optimizer decides how those gradients modify parameters.

```text
Model
  │
  ▼
Gradients
  │
  ▼
Optimizer
  │
  ▼
Parameter Updates
```

---

# Technology Stack

| Technology          | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| Python              | Core implementation                             |
| Native Python Lists | Tensor representation and numerical computation |
| Graphviz            | Computational-graph visualization               |

### Intentionally avoided for core computation

The core tensor and automatic-differentiation implementation does **not** depend on:

* NumPy
* PyTorch
* TensorFlow
* An external automatic-differentiation engine

Tensor operations, shape handling, broadcasting, matrix multiplication, and gradient propagation are implemented using Python itself.

This is an intentional design decision rather than a limitation accidentally introduced by the implementation.

---

# What This Project Demonstrates

The project provides hands-on implementation experience with:

* Reverse-mode automatic differentiation
* Computational graphs
* Chain rule
* Backpropagation
* Gradient accumulation
* Tensor operations
* Broadcasting
* Gradient reduction
* Matrix multiplication
* Linear layers
* Activation functions
* Loss functions
* Mini-batch training
* SGD
* RMSprop
* Adam
* Model persistence
* Computational-graph visualization
* Numerical debugging
* Neural-network training fundamentals

---

# Development Evolution

The project was developed incrementally.

Each stage introduced another abstraction on top of the previous one.

## Phase 1 — Scalar Autograd

Implemented:

```text
Value
  ↓
Operations
  ↓
Computational Graph
  ↓
Backward Pass
```

The objective was to understand automatic differentiation at the smallest practical level.

---

## Phase 2 — Scalar Neural Networks

Neural-network abstractions were built on top of scalar values.

Conceptually:

```text
Neuron
   ↓
Layer
   ↓
MLP
```

This demonstrated how neural networks can be constructed using the same computational-graph mechanism.

---

## Phase 3 — Tensor Engine

The scalar implementation was generalized to multi-dimensional tensors represented using native Python lists.

This introduced additional challenges involving:

* Tensor shapes
* Dimensions
* Matrix multiplication
* Element-wise operations
* Gradient shapes

---

## Phase 4 — Broadcasting

Broadcasting support was added.

The backward pass required gradients to be reduced back to the original operand shapes.

This introduced explicit gradient-unbroadcasting logic.

---

## Phase 5 — Matrix Multiplication

Matrix multiplication and its analytical gradients were implemented and validated.

This enabled the tensor engine to represent the core mathematical operation behind linear neural-network layers.

---

## Phase 6 — Neural-Network Layers

The tensor engine was used to construct trainable linear layers:

$$
y=xW+b
$$

Weights and biases became tensors participating in the computational graph.

---

## Phase 7 — Training

The project evolved into a complete training pipeline:

```text
Dataset
   ↓
Mini-Batches
   ↓
Forward Pass
   ↓
Loss
   ↓
Backward Pass
   ↓
Gradients
   ↓
Optimizer
   ↓
Parameter Update
```

---

## Phase 8 — Multiple Optimizers

The optimization layer was expanded to include:

```text
SGD
RMSprop
Adam
```

This allows different optimization strategies to operate on the same model parameters.

---

# Why Build Autograd From Scratch?

High-level frameworks make deep-learning development significantly easier.

However, the abstraction can hide the mechanics behind operations such as:

```python
loss.backward()
```

and:

```python
optimizer.step()
```

Implementing those mechanisms manually exposes the underlying process:

```text
Mathematical Expression
        ↓
Computational Graph
        ↓
Local Derivatives
        ↓
Chain Rule
        ↓
Reverse Traversal
        ↓
Parameter Gradients
        ↓
Optimization
```

The project therefore focuses on understanding the mechanics rather than simply consuming them through a framework API.

---

# Limitations

This project is intentionally educational and lightweight.

It is **not intended to replace production deep-learning frameworks**.

The current implementation does not attempt to provide the infrastructure found in mature frameworks, including:

* GPU/CUDA execution
* Distributed training
* Automatic mixed precision
* Large-scale data-loading infrastructure
* Production inference optimization
* Hardware-specific kernels
* Large model ecosystems
* Automatic graph compilation
* Distributed optimizers
* Production-grade checkpoint management

These limitations are intentional.

The primary objective is understanding the underlying mechanics of automatic differentiation and neural-network training.

---

# Future Improvements

Potential extensions include:

* Additional tensor operations
* Convolutional layers
* Dropout
* Batch normalization
* Additional optimizers
* Learning-rate schedulers
* Improved module abstractions
* Parameter/state dictionaries
* Numerical gradient-checking utilities
* More comprehensive test coverage
* Improved serialization
* Training metrics
* Dataset abstractions
* DataLoader abstractions
* GPU acceleration
* Computational-graph optimization
* Memory-efficient backward passes

---

# Learning Outcomes

Building this project provided practical understanding of concepts that are often treated as black boxes by high-level machine-learning frameworks.

## Automatic Differentiation

Understanding how a computation graph can be constructed during the forward pass and later traversed backward to calculate derivatives.

## Backpropagation

Understanding that neural-network training fundamentally relies on repeated application of the chain rule.

## Broadcasting

Understanding why forward broadcasting is straightforward while backward propagation requires gradients to be reduced to the original tensor shape.

## Optimization

Understanding the distinction between:

```text
Computing a gradient
```

and:

```text
Using that gradient to update parameters
```

## Tensor Algebra

Understanding how:

* Matrix multiplication
* Transposition
* Element-wise operations
* Broadcasting
* Reduction

interact with neural-network computations.

## Training

Understanding the complete relationship between:

```text
Data
  ↓
Forward Pass
  ↓
Loss
  ↓
Backward Pass
  ↓
Gradients
  ↓
Optimizer
  ↓
Updated Weights
```

---

# Comparison With Modern Frameworks

Conceptually, this project implements a small subset of functionality found in frameworks such as PyTorch.

| Concept               | This Project | Modern DL Frameworks |
| --------------------- | :----------: | :------------------: |
| Computational graph   |       ✓      |           ✓          |
| Reverse-mode autodiff |       ✓      |           ✓          |
| Tensor operations     |       ✓      |           ✓          |
| Broadcasting          |       ✓      |           ✓          |
| Matrix multiplication |       ✓      |           ✓          |
| Backpropagation       |       ✓      |           ✓          |
| Linear layers         |       ✓      |           ✓          |
| Activation functions  |       ✓      |           ✓          |
| SGD                   |       ✓      |           ✓          |
| RMSprop               |       ✓      |           ✓          |
| Adam                  |       ✓      |           ✓          |
| Model persistence     |       ✓      |           ✓          |
| Graph visualization   |       ✓      |        Varies        |
| GPU acceleration      |       —      |           ✓          |
| Distributed training  |       —      |           ✓          |
| Large-scale ecosystem |       —      |           ✓          |

The comparison is about **core concepts**, not feature parity.

Modern frameworks provide significantly broader functionality, optimized numerical kernels, hardware acceleration, and extensive ecosystems.

This project instead prioritizes transparency and understanding.

---

# Project Philosophy

> **Don't just use deep-learning frameworks. Understand what they are doing underneath.**

The project deliberately starts from simple scalar operations and progressively builds toward tensor-based neural-network training.

The progression is:

```text
Scalar Mathematics
        ↓
Automatic Differentiation
        ↓
Computational Graphs
        ↓
Backpropagation
        ↓
Tensor Operations
        ↓
Neural-Network Layers
        ↓
Optimization
        ↓
Mini-Batch Training
        ↓
Model Persistence
```

This progression mirrors the increasing abstraction levels encountered when moving from mathematical definitions to practical machine-learning systems.

---

# Status

The project currently contains:

* Scalar reverse-mode autograd
* Tensor-based reverse-mode autograd
* Native Python-list tensor representation
* Broadcasting support
* Gradient unbroadcasting
* Matrix multiplication
* Gradient propagation
* Gradient accumulation
* Linear neural-network layers
* Activation-function support
* Loss-function support/concepts
* Mini-batch training
* SGD
* RMSprop
* Adam
* Computational-graph visualization
* Model persistence

The architecture remains extensible for additional tensor operations, neural-network components, optimization methods, and execution backends.

---

# Author

**Rahul Bhargav R**

Bachelor of Engineering — Information Science & Engineering
2026

---

[//]: # (# License)

[//]: # ()
[//]: # (Add the repository's chosen license here.)

[//]: # ()
[//]: # (For example:)

[//]: # ()
[//]: # (```text)

[//]: # (MIT License)

[//]: # (```)


