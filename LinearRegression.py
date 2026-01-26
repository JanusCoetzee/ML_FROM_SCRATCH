import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = 0
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias


            # derivative w.r.t. weights
            # Algorithm breakdown:
            # 1. Compute the difference between predicted and actual values (y_predicted - y).
            # 2. Multiply the transpose of the feature matrix (X.T) with this difference
            dw = 1/n_samples * np.dot(X.T, (y_predicted - y)) 

            # derivative w.r.t. bias
            # Algorithm breakdown:
            # 1. Compute the difference between predicted and actual values (y_predicted - y
            # 2. Sum all the differences and scale by the number of samples
            db = 1/n_samples * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
         return np.dot(X, self.weights) + self.bias

