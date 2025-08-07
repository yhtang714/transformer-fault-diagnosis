import numpy as np
from scipy.optimize import minimize

def h_func(W, d):
    E = np.multiply(W, W)
    return np.trace(np.linalg.matrix_power(np.eye(d) + E / d, d)) - d

def loss_function(W, X, lambda1):
    n, d = X.shape
    W = W.reshape(d, d)
    loss = 0.5 / n * np.square(X - X @ W).sum()
    l1 = lambda1 * np.sum(np.abs(W))
    return loss + l1

def objective(flat_W, X, d, lambda1, alpha):
    W = flat_W.reshape(d, d)
    return loss_function(W, X, lambda1) + alpha * h_func(W, d) ** 2

def notears_numpy(X, lambda1=0.01, alpha=10.0):
    d = X.shape[1]
    W0 = np.zeros((d, d)).flatten()
    res = minimize(objective, W0, args=(X, d, lambda1, alpha), method='L-BFGS-B')
    return res.x.reshape(d, d)
