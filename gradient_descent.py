import numpy as np


def gradient_descent(
    theta_0,
    cost_function,
    cost_function_gradient,
    learning_rate=0.01,
    threshold=0.001,
    max_iter=10000,
    params=[]
):
    theta = theta_0
    iteration = 0
    costs = []
    thetas = []

    while np.linalg.norm(cost_function_gradient(theta, *params)) > threshold and iteration < max_iter:
        iteration += 1
        theta -= learning_rate * cost_function_gradient(theta, *params)
        costs.append(cost_function(theta, *params))
        thetas.append(theta.copy())

    return theta, costs, thetas



def regression(X, y, t, cost, grad, step=0.1, n=1000, on_step=None): 
    costs = []
    for i in range(n):
        t -= step * grad(X, y, t)
        costs.append(cost(X, y, t))

        if on_step:
            on_step(t)
    
    return t, costs