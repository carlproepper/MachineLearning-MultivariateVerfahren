import numpy as np
from decision_tree import DecisionTree

class GradientBoosting:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.models = []

    def fit(self, X, y):
        residual = y.copy()
        for _ in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residual)
            predictions = tree.predict(X)
            residual -= self.learning_rate * predictions
            self.models.append(tree)

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for tree in self.models:
            predictions += self.learning_rate * tree.predict(X)
        return predictions
