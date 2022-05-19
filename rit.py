import random
import numpy as np
import pandas as pd


class RITNode(object):
    def __init__(self, val, depth=0):
        self.val = val
        self.children = []
        self.depth = depth

    def add_child(self, node):
        if node and not node.is_empty():
            self.children.append(node)

    def is_empty(self):
        return len(self.val) == 0

    def traverse_depth_first(self):
        yield self
        for child in self.children:
            yield from child.traverse_depth_first()


def get_frequency(val, c, X, freqs):
    val = tuple(sorted(val))
    if c not in freqs:
        freqs[c] = {}
    if val not in freqs[c]:
        X_remain = X
        for s in val:
            f, v = map(lambda x:int(float(x)), s.split('-'))
            X_remain = X_remain[X_remain[:, f] == v]
        freqs[c][val] = len(X_remain) / len(X) if len(X) else 0
    return freqs[c][val]


def generate_rit_sample(X):
    value = X[random.randint(0, len(X) - 1)]
    return ['{}-{}'.format(i, x) for i, x in enumerate(value)]


def generate_tree(X, max_depth, num_splits, noisy_split, depth, parent, c, X0, freqs, theta_0=1):
    val = generate_rit_sample(X)
    if depth:
        val = np.intersect1d(val, parent)
    node = RITNode(val, depth)
    if max_depth > depth and len(val) and get_frequency(val, c, X0, freqs) <= theta_0:
        if noisy_split:
            num_splits += np.random.randint(low=0, high=2)
        for _ in range(num_splits):
            node_child = generate_tree(X, max_depth, num_splits, noisy_split, depth + 1, val, c, X0, freqs, theta_0)
            node.add_child(node_child)
    return node


def generate_trees(X, y, n_tree, max_depth, num_splits, noisy_split, freqs, theta_0):
    trees = {c: [] for c in np.unique(y)}
    for c in trees:
        x_c, x_0 = X[y == c], X[y != c]
        for _ in range(n_tree):
            trees[c].append(generate_tree(x_c, max_depth, num_splits, noisy_split, 0, [], c, x_0, freqs, theta_0))
    return trees


def get_interaction(trees, max_depth):
    res = set()
    for rit_tree in trees:
        for node in rit_tree.traverse_depth_first():
            if node.depth == max_depth:
                res.add(tuple(node.val))
    return res


def select_confident_interactions(interactions, X, y, freqs, theta_0):
    confident_interactions = {
        c: [val for val in interactions[c] if get_frequency(val, c, X[y != c], freqs) <= theta_0]
        for c in interactions
    }
    return confident_interactions


def select_frequent_interactions(interactions, X, y, freqs, theta_1):
    frequent_interactions = {
        c: [val for val in interactions[c] if get_frequency(val, c, X[y == c], freqs) >= theta_1]
        for c in interactions
    }
    return frequent_interactions


def select_interactions(interactions, X, y, freqs_in, freqs_out, n_freq, n_conf):
    n_freq //= len(interactions)
    n_conf //= len(interactions)
    res_interactions = {
        c: sorted(interactions[c], key=lambda val: -get_frequency(val, c, X[y == c], freqs_in))[:n_freq]
        for c in interactions
    }
    res_interactions = {
        c: sorted(interactions[c], key=lambda val: get_frequency(val, c, X[y == c], freqs_out))[:n_conf]
        for c in interactions
    }
    return res_interactions


def genarate_rule(interactions, features):
    def str_to_item(s):
        f, v = map(lambda x:int(float(x)), s.split('-'))
        f = features[f]
        return f, v

    return [(tuple(str_to_item(s) for s in val), c) for c in interactions for val in interactions[c]]


class RIT(object):
    def __init__(self, n_tree=10, max_depth=3, num_splits=5, noisy_split=False,
                 theta_0=1, theta_1=0, n_freq=None, n_conf=None,
                 random_state=2020):
        """Init class with features that need to be encoded"""
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.num_splits = num_splits
        self.noisy_split = noisy_split
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.n_conf = n_conf
        self.n_freq = n_freq
        random.seed(random_state)

    def fit(self, X, y):
        """Get the useful features.

        Args:
            X: input data set, a DataFrame.
            y: labels, an 1d np.array.
        """
        # process the input
        features = list(X.columns)
        X = X.values
        classes = np.unique(y)

        # init
        self.new_features = []
        self.freqs_in = {c: {} for c in classes}
        self.freqs_out = {c: {} for c in classes}

        # generate trees
        trees = generate_trees(X, y, self.n_tree, self.max_depth, self.num_splits, self.noisy_split, self.freqs_out, self.theta_0)
        # extract interactions from the leaf nodes
        interactions = {c: get_interaction(trees[c], self.max_depth) for c in trees}
        # select confident itemsets
        interactions = select_confident_interactions(interactions, X, y, self.freqs_out, self.theta_0)
        # select frequent itemsets
        interactions = select_frequent_interactions(interactions, X, y, self.freqs_in, self.theta_1)
        # select the most frequent and confident itemsets
        interactions = select_interactions(interactions, X, y, self.freqs_in, self.freqs_out, self.n_freq, self.n_conf)
        # generate the rules
        self.rules = genarate_rule(interactions, features)

        # generate feature names
        self.generate_feature_name()

    def transform(self, X):
        """Return the DataFrame of associated features.

        Args:
            X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated features.
        """
        res = {}
        for antecedents, _ in self.rules:
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            if feature_name not in res:
                idx = [True] * len(X)
                for f, val in antecedents:
                    idx &= (X[f]==val)
                res[feature_name] = np.where(idx, 1, 0)
        return pd.DataFrame(res, index=X.index, columns=self.new_features, dtype=np.int8)

    def transform_inter(self, X):
        """Return the DataFrame of interactive features.

        Args:
           X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated interactive effects.
        """
        res = {}
        for antecedents, _ in self.rules:
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            if feature_name not in res and len(antecedents) > 1:
                idx = [True] * len(X)
                for f, val in antecedents:
                    idx &= (X[f]==val)
                res[feature_name] = np.where(idx, 1, 0)
        return pd.DataFrame(res, index=X.index, dtype=np.int8)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for antecedents, _ in self.rules:
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)


class RIT_ctn(RIT):
    """Select interactions of continuous features."""
    def __init__(self, n_tree=10, max_depth=3, num_splits=5, noisy_split=False,
                 theta_0=1, theta_1=0, n_freq=None, n_conf=None,
                 random_state=2020):
        super().__init__(n_tree, max_depth, num_splits, noisy_split, theta_0, theta_1,n_freq, n_conf, random_state)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for antecedents, _ in self.rules:
            if len(antecedents) > 1:
                feature_name = '*'.join([str(f)[:-5] for f, _ in antecedents])
                if feature_name not in self.new_features:
                    self.new_features.append(feature_name)

    def transform(self, X):
        res = {}
        for antecedents, _ in self.rules:
            if len(antecedents) == 1:
                f1 = str(antecedents[0][0])[:-5]
                if f1 not in res:
                    res[f1] = X[f1].values
            else:
                fs = [str(f)[:-5] for f, _ in antecedents]
                feature_name = '*'.join(fs)
                if feature_name not in res:
                    vals = np.ones(len(X))
                    for f in fs:
                        vals *= X[f].values
                    res[feature_name] = vals
        return pd.DataFrame(res, index=X.index)

    def transform_inter(self, X):
        res = {}
        for antecedents, _ in self.rules:
            if len(antecedents) > 1:
                fs = [str(f)[:-5] for f, _ in antecedents]
                feature_name = '*'.join(fs)
                if feature_name not in res:
                    vals = np.ones(len(X))
                    for f in fs:
                        vals *= X[f].values
                    res[feature_name] = vals
        return pd.DataFrame(res, index=X.index)