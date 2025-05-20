import numpy as np
from collections import Counter

class DecisionTreeScratch:
    def __init__(self, max_depth=4, min_samples_split=2, min_samples_leaf=1, min_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_features = min_features
        self.tree = None

    def entropy(self, y):
        counts = np.bincount(y)
        probs = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def info_gain(self, y, left_y, right_y):
        parent_entropy = self.entropy(y)
        n = len(y)
        return parent_entropy - (
            len(left_y) / n * self.entropy(left_y) + len(right_y) / n * self.entropy(right_y)
        )

    def best_split(self, X, y):
        n_features = X.shape[1]
        features = np.arange(n_features)

        if self.min_features is not None and self.min_features < n_features:
            features = np.random.choice(features, self.min_features, replace=False)

        best_gain, best_feat, best_thresh = 0, None, None

        for feat in features:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = X[:, feat] > thresh
                left_y, right_y = y[left_mask], y[right_mask]

                if len(left_y) < self.min_samples_leaf or len(right_y) < self.min_samples_leaf:
                    continue

                gain = self.info_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh
        return best_feat, best_thresh

    def build(self, X, y, depth):
        if (
            depth >= self.max_depth or
            len(np.unique(y)) == 1 or
            len(y) < self.min_samples_split
        ):
            return Counter(y).most_common(1)[0][0]

        feat, thresh = self.best_split(X, y)
        if feat is None:
            return Counter(y).most_common(1)[0][0]

        left_mask = X[:, feat] <= thresh
        right_mask = ~left_mask

        left = self.build(X[left_mask], y[left_mask], depth + 1)
        right = self.build(X[right_mask], y[right_mask], depth + 1)

        return {'feature': feat, 'threshold': thresh, 'left': left, 'right': right}

    def fit(self, X, y):
        self.tree = self.build(np.array(X), np.array(y), 0)

    def predict_one(self, x, node):
        while isinstance(node, dict):
            if x[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node

    def predict(self, X):
        return np.array([self.predict_one(x, self.tree) for x in np.array(X)])
