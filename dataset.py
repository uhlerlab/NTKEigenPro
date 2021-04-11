import numpy as np


def f(x):
    return x**2 + 2*x + 1


def make_data():
    n = 100
    d = 10
    f_v = np.vectorize(f)

    X = np.random.randn(n, d)
    y = np.random.randn(n, 1)


    return X.astype('float32'), y.astype('float32')
