# Artificial Intelligence for Aerospace Engineers

## Lecture 1 (Introduction)

- Machines are great at finding correlations; but bad **interpretability** and **robustness** to data
- Humans are good at finding principles; don't forget about the causes/approaches available when solving problems!

Admin:
- 17h lectures, 2h computing, 5h computing tutorials
- 70% final exam, 30% weekly assignments (need to pass 4/6 minimum; pass fail 3 attempts autograder)
- MCQ, 1.5h, theory and good practice (no coding exercises); weekly quizzes designed similarly

Definitions:
- Artificial intelligence: study and design of intelligent agents
- Machine learning: tool that can use data (inputs) to give a prediction (outputs)
- Data science and data mining: statistical inference, data visualization, communication; extract knowledge from data
- Deep learning: multilayered NNs learn from vast amounts of data

Why AI in Aerospace Engineering:
- Vast and increasing data
- Advances in HPC (Moore's law)
- Improvements to sensing technologies, data storage and transfer
- Scalable algorithms from statistics and applied mathematics
- Considerably investment by industry, leading to an abundance of open source software
- Digital thread and digital twin: digitalization every stage of design/manufacture of product, collect data at every stage (digital thread), very accurate model of product (digital twin)

Types of AI Algorithms:
- Supervised learning: describe features and labels for data (e.g. regression)
- Unsupervised learning: describe data, algorithm to cluster data according to some similarity (define its own features to separate data into classes)
- Semi-supervised learning: inbetween, such as **reinforcement learning** 

Machine learning workflow:
- Data: acquisition, analysis, pre-processing
- Model: design, adjustment/tuning
- Training: optimisation, evaluation
- Deployment

## Lecture 2 (Linear regression)

### One variable linear regression

The goal of regression: find `f` such that `f(x)` approximates `y`. Regression analysis consists of the following steps:

1. Select a **hypothesis function** `h(x)` which (we assume) models the data. Here, we choose `h(x)` to be linear, assume that `y` = `h(x)` = `\theta_0 + \theta_1 * x`. Where `\theta_0` is the intercept, `\theta_1` is the slope to be found
2. **Loss function** measures the error between the data points `y_i` and the predictions from our model `h(x_i)`. We choose the mean-squared error (MSE) to globally measure the quality of the predictions

![](img/mse-equation.PNG)

3. **Training**/*Learning*. Find the two parameters of the linear model `\theta_0` and `\theta_1` for the loss function to be minimised. This optimal set of parameters corresponds to the "best" linear model: an optimisation problem

### Gradient Descent

To solve the minimisation problem, start at a point, find the direction of maximum decrease of the loss function, then take a step in that direction, and iterate until the loss function cannot decrease anymore.

1. **Choose a starting point**. Choose an initial guess of `\theta_0` and `\theta_1`, normally done by randomly guessing.
2. **Calculate the gradient**. Calculate the gradient of the loss function with respect to an infinitesimal change in your parameters `\theta_0` and `\theta_1`. This will give the direction along which the loss function increases the most:

![](img/grad-descent-1.png)

3. **Update**. Take a small step in the direction of the most negative gradient (largest descent) and update your parameters `\theta_0` and `\theta_1`.

![](img/grad-descent-2.png)

![](img/grad-descent-3.png)

4. **Repeat**. The loss function MSE is quadratic; gradient changes linearly. Therefore, we need to update the gradient at the new point, and repeat steps 1-3 until the variation of the cost function is zero (to *numerical tolerance*) and/or a *maximum number of iterations* is reached. Both of these training parameters are user-defined.

Example of calculating gradient descent by hand:

![](img/grad-descent-4.png)

### Multi-variable linear regression

For multi-variable functions, we generalise to `N` features. The steps taken are similar to the one-variable case, except we generalise to `N` features `x_1 ... x_N`, and `N+1` parameters `\theta_0 ... \theta_N`.

1. **Hypothesis**. Select a multi-variable linear hypothesis function:

![](img/multi-descent-1.png)

2. **Loss function**. Mean-squared error (MSE):

![](img/multi-descent-2.png)

3. **Training**. Find the `N+1` parameters of the linear model to minimise loss `J`

As before, gradient descent consists of the same steps:

![](img/multi-descent-4.png)

Example of calculating multi-variable gradient descent by hand:

![](img/multi-descent-5.png)

The path taken by gradient descent can be far more complicated:

![](img/grad-descent-5.png)

### Vectorization

Vectorization enables a compact notation, which is easy to program in computer code and faster to execute. This is why linear algebra is so widely used in machine learning.

![](img/vectorization-1.png)
![](img/vectorization-2.png)

### Regularization

Loss (in this case MSE) measures the *training error*, which is only relevant to the train set. The error between the prediction on unseen features and the test set is *generalization error*. A small training error is not sufficient; it may:

* **Underfit** when training error is too large; model has a high bias. Possibly too few parameters/*capacity*.
* **Overfit** when training error is small; but generalization error is large. *Generalization gap* is large; model has a high variance. Possibly too many parameters/*capacity*.

![](img/model-error.png)

A good machine learning model should have a good *bias-variance trade off*. This can be achieved by minimising the generalisation error, which is a sum of the (squared) bias and the variance. This can be practically achieved by **regularizing** the problem. Add an extra term in the loss function with regularization parameter `\lambda`:

![](img/regularization-1.png)

If `\lambda` = 0, problem is not regularized.
If `\lambda` is small, large values of `\sum \theta_j ^2` are allowed; model might contain too many parameters; might overfit.
If `\lambda` is large, only small values of `\sum \theta_j ^2` are allowed; training will reduce the number of parameters; might underfit.

`\lambda` has a typical range of values between `1e-6` to `1e-2` but is strongly problem dependent and difficult to generalise.

### Learning rate and feature scaling

**Learning rate**. Small values of `lr` make gradient descent converge to an accurate minimum, but may be time consuming. Large values of `lr` might make gradient descent converge to an inaccurate minimum / diverge, but computation will be cheap. Adaptive learning rates are possible with the [*Adaptive Moment Estiation* (ADAM) optimiser](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/#:~:text=Adam%20is%20a%20replacement%20optimization,sparse%20gradients%20on%20noisy%20problems), which uses the information of previous gradients to accelerate convergence. Typical values are `0.001 - 1` usually in step multiples of 3.

**Feature scaling**. Sometimes features have different absolute values; the larger variable will have a larger variation with respect to the parameters `\theta_N`, thus will be much more heavily influenced by that feature. Convergence in the direction of other features is slow. The algorithm will converge faster if the range of the feature has the same order of magnitude. We scale the values to standardise each feature:

![](img/feature-scaling-1.png)

where `^` denotes normalisation, `\mu_j` is the mean value and `\sigma_j` is the standard deviation of `x_j`.

### Variations of gradient descent

* **Batch gradient descent**. What we have been looking at thus far. The whole training dataset (*batch*) is used at each iteration. The gradient points in the steepest gradient direction, but the algorithm is prone to overfitting and needs large memory requirements for large datasets.
* **Stochastic gradient descent**. Only one data point is selected at random to be used at each iteration. Gradient may not point in the steepest descent direction, progress may be noisy, convergence may be slow, less prone to overfitting.
* **Mini-batch gradient descent**. A hybrid approach between the previous two. The training set is divided into small batches, gradient is computed in each *mini-batch*. Algorithm is quick and can be easily parallelised in computer software because each mini-batch gradient computation is independent; results may depend on batch size, less prone to overfitting.

![](img/alternate-gradient-descent.png)

Additionally, training data can be re-shuffled randomly a number of times and repeated in *epochs*. This minimises the importance of the last seen datapoint and improves convergence. In most ML libraries, mini-batch gradient descent is applied, where you can select a custom batch size.