# Summary of exercises

This provides a "table of contents" of the exercises in each set of lecture notes for easy lookup.

## Linear regression (Lecture 2)

1. Show why the gradient is "the direction and rate of maximum increase of a function".
2. Given a differentiable convex function `f`, the graph lies above all of its tangent planes.
   1. Give three examples of convex functions.
   2. Show that a local minimum of a strictly convex function is a global minimum.
   3. If you find a minimum of the mean-squared error of *linear regression*, is it the global minimum?
   4. Is the sum of two convex functions convex?
3. One dimensional gradient descent (link provided):
   1. Explain the behaviour of gradient descent with Initial slope = 0.5; Learning rate = 0.14; Batch size = 20; Number of iterations = 25. Suggest strategies to improve convergence.
4. The mean-squared error can be written in matrix form as `J` = ... Can you find the optimal set of parameters `\theta` without gradient descent?

Assignment 1: Multivariate linear regression in MATLAB.

## Logistic regression (Lecture 3)

1. Entropy gives the average level of information of the outcomes. Which of the following datasets of two classes 1 and 2 has maximum entropy?
2. Calculate the precision, recall and accuracy.

1. What is the geometric interpretation of the Lagrangian in Eq 26?
2. Derive the Lagrangian Dual in Eq 28.
3. Given the polynomial kernel, find `\phi` such that `K` = `\phi_x` . `\phi_y`. Assume that the vectors are two dimensional.

Assignment 2: Kernel trick for SVMs in MATLAB.

## Feedforward neural networks (Lecture 4)

1. How many input paths from the input layer to node 7 (nodes per layer**layers)
2. Learn tensorflow in python3

### Tutorial

1. How to use Tensorflow/Keras to make a basic classifier.

## Convolutional neural networks (Lecture 5)

1. How many parameters would the following CNN have with and without parameter sharing?
2. Read about the following CNN architectures (AlexNet, GoogleNet, VGGNet, ResNet).


## Clustering (Lecture 7)

1. Clustering interactive [website](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
2. Gaussians interactive [website](https://distill.pub/2019/visual-exploration-gaussian-processes/)