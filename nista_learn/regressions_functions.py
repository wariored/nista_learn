import numpy as np
import pandas as pd


def parameters_initializer(shape):
    """initialize weight and a ones array"""
    theta = np.zeros((1, shape[1] + 1))
    ones = np.ones([shape[0], 1])
    return theta, ones


def cost_function(loss, m):
    return np.sum(loss ** 2) / (2 * m)


def compute_gradient_linear(x, y, iterations=1000, learning_rate=0.001, show=False, normalize=False):
    m = x.shape[0]
    theta, ones = parameters_initializer(x.shape)
    x = np.concatenate((ones, x), axis=1)
    if normalize:
        x = x / x.mean(axis=0)
        y = y / y.mean(axis=0)
    y = y.reshape((-1, 1))

    theta_transpose = theta.T

    cost_list = []
    for i in range(1, iterations + 1):
        hypothesis = np.dot(x, theta_transpose)
        loss = hypothesis - y
        cost = cost_function(loss, m)
        gradient = np.dot(x.T, loss) / m
        theta_transpose = theta_transpose - learning_rate * gradient

        if i % 100 == 0:
            if show:
                print(f"After {i} iterations | cost: {cost}")
            cost_list.append(cost)
        if cost <= 0.000000001:
            break
    return theta_transpose, cost_list


def sigmoid(z):
    """return the sigmoid value of an input z"""
    sigma = 1 / (1 + np.exp(-z))
    return sigma


def compute_gradient_logistic(x, y, iterations=1000, learning_rate=0.001, show=False):
    m = x.shape[0]
    theta, ones = parameters_initializer(x.shape)
    x = np.concatenate((ones, x), axis=1)
    y = y.reshape((-1, 1))
    theta_transpose = theta.T

    cost_list = []
    for i in range(1, iterations + 1):
        hypothesis = np.dot(x, theta_transpose)
        hypothesis = sigmoid(hypothesis)
        loss = hypothesis - y
        cost = cost_function(loss, m)
        gradient = np.dot(x.T, loss) / m
        theta_transpose = theta_transpose - learning_rate * gradient

        if i % 100 == 0:
            if show:
                print(f"After {i} iterations | cost: {cost}")
            cost_list.append(cost)
        if cost <= 0.000000001:
            break
    return theta_transpose, cost_list


def concatenate_with_ones_vector(x):
    ones = np.ones([x.shape[0], 1])
    return np.concatenate((ones, x), axis=1)


def score(y_a, y_b):
    return None
