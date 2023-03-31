import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def cost_function(theta, x, y):
    m = len(y)
    z = np.dot(x, theta)
    h = sigmoid(z)
    cost = -1/m * (np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
    return cost[0]

def gradient(theta, x, y):
    m = len(y)
    z = np.dot(x, theta)
    h = sigmoid(z)
    grad = 1/m * np.dot(x.T, (h - y))
    return grad

def train(x, y, alpha=0.01, num_iters=1000):
    m, n = x.shape
    x = np.hstack((np.ones((m, 1)), x))
    y = y.reshape((-1, 1))
    theta = np.random.rand(n+1, 1)
    for i in range(num_iters):
        theta -= alpha * gradient(theta, x, y)
    return theta