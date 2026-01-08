import numpy as np

def generate_data(n=80,noise = 0.15):
    np.random.seed(42)
    X = np.linspace(-3,3,n).reshape(-1,1)
    y = np.sin(X).ravel() + np.random.normal(0,noise,n)
    return X,y