import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features="sqrt"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            indices = np.random.choice(len(X), size=len(X), replace=True)
            sample_X, sample_y = X[indices], y[indices]
            tree = DecisionTree(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(sample_X, sample_y)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
