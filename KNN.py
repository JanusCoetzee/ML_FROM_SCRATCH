# KNN.py
"""
Docstring for KNN
Classifier with Euclidean and Jaccard distance metrics.

"""

import numpy as np
from collections import Counter


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def jaccard_distance(a, b):
    # Check if data is binary (only contains 0s and 1s)
    is_binary = np.all(np.isin(a, [0, 1])) and np.all(np.isin(b, [0, 1]))
    
    if is_binary:
        # Traditional Jaccard for binary sets
        intersection = np.sum(np.logical_and(a, b))
        union = np.sum(np.logical_or(a, b))
    else:
        # Generalized Jaccard for continuous/count data
        intersection = np.sum(np.minimum(a, b))
        union = np.sum(np.maximum(a, b))
    
    # Avoid division by zero
    if union == 0:
        return 0.0
    
    return 1 - intersection / union

class KNN:
    def __init__(self, k=3, distance_metric='euclidean') -> None:
        self.k = k
        if distance_metric == 'euclidean':
            self.distance = euclidean_distance
        elif distance_metric == 'jaccard':
            self.distance = jaccard_distance
        else:
            raise ValueError("Unsupported distance metric")

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict (self,x):
        # compute distances
        distances = [self.distance(x, x_train) for x_train in self.X_train]

        #get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]