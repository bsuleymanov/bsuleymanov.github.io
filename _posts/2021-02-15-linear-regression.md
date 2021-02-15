---
layout: post
title: ML Course. Lesson 1. Linear regression.
subtitle: Linear regression
tags: [test]
comments: false
---

Univariate linear regression: $y = ax + b$, where $x$ is a covariate, $y$ is a target, $a$ is a slope and $b$ is a intercept.

Let's generate some data with underlying function $f(\mathbf{x}) = 3x_1 + 4x_2 + 5x_3 + 2$ and $y = f(\mathbf{x}) + \epsilon$, $\epsilon \sim \mathcal{N}(0, 1)$:
```python
N, M = 100, 3
X = np.random.randn(N, M)
X = np.concatenate([np.ones((N, 1)), X], axis=1)
weights_true = np.array([[2, 3, 4, 5]])
y = X @ weights_true.T + np.random.randn(N, 1)
```
To find weights of linear regression we can use normal equations $\pmb{\hat{\theta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top\mathbf{y}$:
```python
weights_estimated = np.linalg.inv(X.T @ X) @ X.T @ y
```
