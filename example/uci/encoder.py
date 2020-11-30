import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from minheap import MinHeap


class OneHotEncoder(object):
    """One-hot encode the features
    All the values that appear only in test set but not in training set will be regarded as a same default value.

    Attributes:
        features: the (sparse) features that need to be encoded.
        values: a dict whose key is the feature names and value is the values occurs in the train set.
        new_features: the new feature names after one-hot coding.
        indices: a dict whose key is the new feature names and value is the value is the indices in new_features.
    """

    def __init__(self, features):
        """Init class with features that need to be encoded"""
        self.features = features
        self.values = {}
        self.new_features = []
        self.indices = {}

    def fit(self, X):
        """Fit a transformer on the data set X
        Args:
            X: the data set X need to be encoded.
        """
        self.new_features = []
        index = 0
        for f in self.features:
            self.values[f] = X[f].unique()
            for x in self.values[f]:
                self.new_features.append(f + str(x))
                self.indices[f + str(x)] = index
                index += 1

    def transform(self, X):
        """Encode the data set X and return the result
        Args:
            X: the data set X need to be encoded.

        Returns:
            A DataFrame whose columns are the features after encoding.
            All values that appear only in test set but not in training set will be regarded as a same default value
            (actually no values in the result DataFrame will be changed to 1 if an unseen value occurs).
        """
        columns = X.columns
        res = np.zeros((len(X), len(self.new_features)))
        for x, row in zip(X.values, res):
            for i, val in enumerate(x):
                if columns[i] + str(val) in self.indices:
                    row[self.indices[columns[i] + str(val)]] = 1
        return pd.DataFrame(res, columns=self.new_features, index=X.index)

    def fit_transform(self, X):
        """fit a transformer on the data set X and
        return the result of encoding X
        Args:
            X: the data set X need to be trained on and encoded.
        Returns:
            A DataFrame whose columns are the features after encoding.
        """
        self.fit(X)
        return self.transform(X)


class LabelEncoder(object):
    """Label encode the features
    All the values that appear only in test set but not in training set will be regarded as a same default value

    Attributes:
        features: the (sparse) features that need to be encoded.
        encode: a dict whose key is the original value and val is the corresponding code.
        decode: a dict whose key is the feature names and value is another dict
                whose key is a code (integer) and value is the original value.
        scale: bool, whether the labeled features need scaling
        scaler: a dict of MinMaxScaler for each encoded feature
    """

    def __init__(self, features, scale=True):
        """Init class with features that need to be encoded"""
        self.features = features
        self.encode = {}
        self.decode = {}
        self.scale = scale
        self.scaler = {}

    def fit(self, X):
        """Fit a transformer on the data set X"""
        for f in self.features:
            self.scaler[f] = MinMaxScaler()
            if not np.issubdtype(X[f].values.dtype, np.number):
                self.encode[f] = {x: i for i, x in enumerate(X[f].unique())}
                self.decode[f] = {i: x for x, i in self.encode[f].items()}
                self.scaler[f].fit([[0],[len(X[f].unique())]])
            else:
                self.scaler[f].fit(X[f].values.reshape(-1, 1))

    def transform(self, X):
        """Encode the data set X and return the result

        Returns:
            A DataFrame whose columns are the giving features.
            All values that appear only in test set but not in training set will be regarded as a same default value
            (actually no values in the result DataFrame will be changed to 1 if an unseen value occurs).
        """
        res = {}
        for f in self.features:
            if f in self.encode:
                func = lambda x: self.encode[f][x] if x in self.encode[f] else len(self.encode[f])
                res[f] = [func(x) for x in X[f]]
            else:
                res[f] = X[f].values
        X_lbe = pd.DataFrame(res, index=X.index)
        if self.scale:
            for f in self.features:
                X_lbe[f] = self.scaler[f].transform(X_lbe[f].values.reshape(-1,1)).ravel()
        return X_lbe

    def fit_transform(self, X):
        """fit a transformer on the data set X and
        return the result of encoding X
        """
        self.fit(X)
        return self.transform(X)

    def reverse(self, f, i):
        """return the original value of the corresponding code."""
        return self.decode[f][i]


def entropy(count):
    """Calculate the entropy.

    Args:
        count: a dict whose key is an element in the set and value is its frequency of occurrence.
               The sum of the frequencies should be larger than 0.
    Returns:
        the entropy.
    """
    l = sum([count[c] for c in count])
    ent = 0
    for c in count:
        p = count[c] / l
        if p:
            ent -= p * math.log(p)
    return ent


def gain(count1, count2):
    """Calcutlate the information entropy after split a set into two parts.

    Args:
        count1: a dict whose key is an element in the first part after splitting,
                and value is its frequency of occurrence.
        count2: a dict whose key is an element in the second part after splitting,
                and value is its frequency of occurrence.
    Returns:
        the information entropy after splitting.
    """
    l1 = sum([count1[c] for c in count1])
    l2 = sum([count2[c] for c in count2])
    return -(entropy(count1) * l1 + entropy(count2) * l2) / (l1 + l2)


def best_threshold(y, candidates):
    """Find the best threshold dividing the set by which can lead to the best information gain.

    Args:
        y: the labels sorted by a corresponding feature.
        candidates: a list of thresholds(index) from which we should find the best one.
    Returns:
        the best threshold and the corresponding information gain.
    """
    count1 = Counter(y)
    count2 = {c: 0 for c in count1}
    best_thres, best_ig = 0, -entropy(count1)
    for i in range(len(candidates) - 1):
        count = Counter(y[candidates[i]:candidates[i + 1]])
        for c in count:
            count1[c] -= count[c]
            count2[c] += count[c]
        ig = gain(count1, count2)
        if ig > best_ig:
            best_ig = ig
            best_thres = candidates[i + 1]
    return best_thres, entropy(Counter(y)) - best_ig


class SplitEncoder(object):
    """Discretize the numeric features by information gain.

    Attributes:
        features: the name of numeric features.
        n_class: the number of class after splitting.
        n_candidate: the number of thresholds for each searching.
        thresholds: a dict whose key is a features name and
            the value is the corresponding thresholds.
        new_features: the names of features after splitting.
    """

    def __init__(self, features, n_class, n_candidate=float('inf')):
        """Init class."""
        self.features = features
        self.n_class = n_class
        self.n_candidate = n_candidate
        self.thresholds = {}
        self.new_features = None

    def split_by_entropy(self, x, y, sort=True):
        """Find the best threshold for a feature x and corresponding labels y.

        Args:
            x: a feature.
            y: the labels.
            sort: whether the feature and labels have to be sorted.

        Returns:
            the best threshold and the corresponding information gain.
        """
        if sort:
            sort_index = sorted(range(len(x)), key=lambda i: x[i])
            y_sort = y[sort_index]
        else:
            y_sort = y
        n_candidate = min(self.n_candidate, len(y))
        candidates = [i * len(y) // n_candidate for i in range(n_candidate)]
        best_thres, best_ig = best_threshold(y_sort, candidates)
        return best_thres, best_ig

    def split(self, x, y):
        """Find (n_class-1) best thresholds for a feature x and corresponding labels y.

        Args:
            x: a feature.
            y: the labels.

        Returns:
            a list of the best thresholds(x-value).
        """
        heap = MinHeap(self.n_class)
        sort_index = sorted(range(len(x)), key=lambda i: x[i])
        x_sort = x[sort_index]
        y_sort = y[sort_index]
        thres_i, ig = self.split_by_entropy(x_sort, y_sort, sort=False)
        heap.push((tuple(range(len(y))), thres_i), -ig)
        while heap.size < self.n_class and heap.minimum() < 0:
            (piece, thres), ig = heap.pop()
            piece1, piece2 = list(piece[:thres]), list(piece[thres:])
            thres_i1, ig1 = self.split_by_entropy(x_sort[piece1], y_sort[piece1], sort=False)
            thres_i2, ig2 = self.split_by_entropy(x_sort[piece2], y_sort[piece2], sort=False)
            heap.push((piece1, thres_i1), -ig1)
            heap.push((piece2, thres_i2), -ig2)
        return x_sort[sorted([item[0][0][-1] for item in heap.items()])[:-1]]

    def fit(self, X, y):
        """Find the thresholds for every numeric feature."""
        for f in self.features:
            self.thresholds[f] = self.split(X[f].values, y)

    def transform(self, X):
        """Transform a input matrix to its discrete form."""
        res = {}
        for f in self.features:
            res[str(f) + "_disc"] = np.zeros(len(X))
            x = X[f]
            for i, thres in enumerate(self.thresholds[f]):
                res[str(f) + "_disc"][x > thres] = i + 1
        self.new_features = list(res.keys())
        return pd.DataFrame(res, index=X.index)
