import numpy as np


def s2(X):
    x_mean = np.mean(X)
    return float(np.sum(np.square(X - x_mean)))

def cov(X, Y):
    x_mean = np.mean(X)
    y_mean = np.mean(Y)
    return (X - x_mean).T @ (Y - y_mean)
    
def r(X, Y):
    return cov(X, Y) / np.sqrt(s2(X) * s2(Y))
