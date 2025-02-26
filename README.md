# The Perceptron


1. Introduction to Perceptrons
2. Mathematical Foundations
3. Perceptron Learning Algorithm
4. Implementation Details
5. Limitations and Extensions

---

1. __Introduction to Perceptrons__

A perceptron is the simplest type of artificial neural network, introduced by Frank Rosenblatt in 1958. It serves as a binary classifier, deciding whether an input belongs to one class or another.

Key points:
- The perceptron models a single neuron.
- It takes multiple inputs, applies weights to them, sums them up, and passes the result through an activation function (usually a step function).
- Outputs are binary (0 or 1).


Real-world analogy:
Think of the perceptron like a decision-making process:
"If the sum of weighted inputs is above a certain threshold, then activate (1), else don’t activate (0)."

---

2. __Mathematical Foundations__


The perceptron can be represented by:
$$
y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)
$$
- $$x_i$$: Input features
- $$w_i$$: Weights associated with each input
- $$b$$: Bias term (shifts the decision boundary)
- $$f(⋅)$$: Activation function (usually the Heaviside step function)


Step Function:
$$
f(z) = \begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{otherwise}
\end{cases}
$$


Geometrical Interpretation:

    For 2D inputs, the perceptron decision boundary is a line; for 3D, it’s a plane; and in higher dimensions, a hyperplane.

---

3. __Perceptron Learning Algorithm__

The goal: Adjust weights $$w_i$$ and bias $$b$$ to minimize classification error.
Algorithm Steps:

    Initialize weights $$w_i$$ and bias $$b$$ (often to small random values).
    For each training sample ($$x,y$$):
        Compute the output: $$ŷ=f(w⋅x+b$$).
        Update weights and bias if there is an error:

        $$w_i:=w_i+η(y−ŷ)x_i$$
        $$b:=b+η(y−ŷ)$$

η is the learning rate.

Convergence Theorem:

If the data is linearly separable, the perceptron learning algorithm is guaranteed to converge to a solution.

---


4. __Implementation Details__

Students can implement the perceptron in Python using NumPy. Key aspects:

    Dataset: Use a linearly separable dataset (e.g., AND/OR functions).
    Activation Function: Step function.
    Training Loop: Iterate until the model classifies all points correctly or after a set number of epochs.
    Visualization: Plot decision boundaries for better understanding.

Example datasets:

    AND gate: Linearly separable.
    XOR gate: Not linearly separable (perceptron fails here).

---

5. __Limitations and Extensions__
Limitations:

    Cannot solve non-linearly separable problems (e.g., XOR problem).
    Only suitable for binary classification.
    Step function is not differentiable, limiting gradient-based optimization.

Extensions:

    Multi-layer Perceptron (MLP): Stacking perceptrons with non-linear activation functions (e.g., sigmoid, ReLU).
    Backpropagation: Enables multi-layer training via gradient descent.
    Kernel Trick: Extends perceptrons to non-linear classification.
