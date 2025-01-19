import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        if len(y) < self.min_samples_split or depth == self.max_depth or len(np.unique(y)) == 1:
            return {"label": self._most_common_label(y)}

        feature, threshold = self._best_split(X, y)
        if feature is None:
            return {"label": self._most_common_label(y)}

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_indices], y[left_indices], depth + 1),
            "right": self._build_tree(X[right_indices], y[right_indices], depth + 1),
        }

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if "label" in node:
            return node["label"]
        if x[node["feature"]] <= node["threshold"]:
            return self._traverse_tree(x, node["left"])
        return self._traverse_tree(x, node["right"])

    # Hilfsfunktionen (_best_split, _gini, etc.) werden hier implementiert.
