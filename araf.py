import pandas as pd
import numpy as np


class MinHeap(object):
    """a heap designed for top K problem.
    Attributes:
        capacity: the max size of the heap as well as the number K in top K problem.
        size: the number of the elements in the heap.
        heap: the heap itself, the elements of which are lists consists of items like [key, val],
              If keys are ignored and only vals are considered, the heap is a min heap.
    """
    def __init__(self, capacity):
        """Init the heap."""
        self.capacity = capacity
        self.size = 0
        self.heap = [[None, -float('inf')]] * capacity

    def min_heapify(self, index, key, val):
        """Set the index'th element with [key, val],
        then modify the heap to make it still a min heap.
        """
        while index > 0:
            p = (index-1)//2
            if self.heap[p][1] > val:
                self.heap[index] = self.heap[p].copy()
                index = p
            else:
                break
        self.heap[index] = [key, val]

    def push(self, key, val):
        """Push a new element [key, val] into the heap.
        If the heap is full, the original root will be deleted.
        """
        if self.size < self.capacity:
            self.min_heapify(self.size, key, val)
            self.size += 1
        elif self.minimum() < val:
            self.pop()
            self.push(key, val)

    def pop(self):
        """Delete the root of the heap,
        then modify the heap to make it still a min heap.
        Returns:
             the element at root, whose val is minimum of the heap.
        """
        ans = self.heap[0]
        if self.size:
            self.size -= 1
            index = 0
            while 2*index+1 < self.size:
                a, b = 2*index+1, 2*index+2
                if b < self.size and self.heap[b][1] < self.heap[a][1]:
                    a = b
                if self.heap[a][1] > self.heap[self.size][1]:
                    break
                self.heap[index] = self.heap[a].copy()
                index = a
            self.heap[index] = self.heap[self.size].copy()
        return ans

    def items(self):
        """Return the items in the heap."""
        return self.heap[:self.size]

    def minimum(self):
        """Return the minimum of the heap."""
        return self.heap[0][1] if not self.isempty() else float('inf')

    def isempty(self):
        """Return whether the heap is empty."""
        return self.size == 0

    def reset(self):
        """Reset the heap."""
        self.size = 0


class ARAF(object):
    """Get main effects and interactive features by modified Apriori.
    Attributes:
        n_freq: the number of the frequent itemsets, an integer.
        n_conf: the number of the confident rules, an integer.
        sample_size: if an integer， the size of subsample for selecting frequent sets;
                     if none, use the complete data set.
        _count_1: a dict in the form of {1-itemset: frequency}.
        _count_2: a dict in the form of {2-itemset: frequency}.
        frequent_set: the frequent itemsets for different classes,
            a dict in the form of {label: orderlist},
            where the elements in the orderlist is (item, freq),
            item is in the form of ((index, value),...),
            (index, value) stands for the item "X[index]==value",
            freq is the corresponding frequency, a float.
        confident_rule: the confident rules for different classes,
            an orderlist with elements as ((item, c), conf),
            a dict in the form of {label: orderlist},
            where the elements in the orderlist is (item, conf),
            item is in the form of ((index, value),...),
            (index, value) stands for the item "X[index]==value",
            conf is the corresponding confidence, a float.
        rules: a list of rules, with elements in the form of (antecedents, consequence),
            where antecedents is a tuple with elements in the form of (feature, value),
            (feature, value) stands for the item "X[feature]==value",
            consequence is in the form of (target, c).
        new_features: a list of feature names of the generated features.
    """

    def __init__(self, n_freq, n_conf, sample_size=None):
        """Init class with the parameters"""
        self.n_freq = n_freq
        self.n_conf = n_conf
        self.sample_size = sample_size
        self._count_1 = {}
        self._count_2 = {}
        self.new_features = []
        self.frequent_set = {}
        self.confident_rule = {}
        self.rules = None

    def count_1_itemsets(self, X, y, miss_val):
        """Count 1-itemsets.
        Args:
            X: input data set, an n*p np.array.
            y: labels, an 1*n np.array.
            miss_val: a list of missing values.
        """
        self._count_1 = {c: {} for c in np.unique(y)}
        for row, c in zip(X, y):
            for f in range(len(row)):
                if row[f] not in miss_val:
                    item = ((f, row[f]),)
                    if item in self._count_1[c]:
                        self._count_1[c][item] += 1
                    else:
                        self._count_1[c][item] = 1

    def select_frequent_1_itemsets(self):
        """Select 1-itemsets that are most frequent for each class."""
        for c in self._count_1:
            for item in self._count_1[c]:
                self.frequent_set[c].push(item, self._count_1[c][item])

    def count_2_itemsets(self, X, y):
        """Count 2-itemsets.
        Args:
            X: input data set, an n*p np.array.
            y: labels, an 1*n np.array.
        """
        classes = np.unique(y)
        self._count_2 = {c: {} for c in classes}
        frequent_set = {c: [items[0] for items, _ in self.frequent_set[c].items()] for c in classes}
        freq_columns = {c: sorted({item[0] for item in frequent_set[c]}) for c in classes}
        # print(freq_columns)
        for row, c in zip(X, y):
            for i1 in range(len(freq_columns[c])-1):
                f1 = freq_columns[c][i1]
                val1 = row[f1]
                if (f1, val1) in frequent_set[c]:
                    # print(f1, val1)
                    for i2 in range(i1 + 1, len(freq_columns[c])):
                        f2 = freq_columns[c][i2]
                        val2 = row[f2]
                        # print(f1, f2, val1, val2)
                        if (f2, val2) in frequent_set[c]:
                            items = ((f1, val1), (f2, val2))
                            if items in self._count_2[c]:
                                self._count_2[c][items] += 1
                            else:
                                self._count_2[c][items] = 1

    def select_frequent_itemsets(self):
        """Select 1-itemsets and 2-itemsets that are most frequent for each class."""
        for c in self._count_2:
            for item in self._count_2[c]:
                self.frequent_set[c].push(item, self._count_2[c][item])

    def select_confident_rule(self, X, y, prior):
        """Select frequent 1-itemsets and 2-itemsets that are most confident.
           If several itemsets have the same confidence and space is not sufficient to keep all of them,
           then only those with largest support will be remained.
           If a main effect and its interaction are both confident,
           then the interaction will be remained only when it's more confident than the main effect.
        """

        def calculate_confidence(X, y, item, c):
            indices = np.arange(len(X))
            for i, v in item:
                indices = indices[X[indices, i] == v]
            return sum(y[indices] == c) / len(indices)

        def better_interact(item, conf, confident_rules):
            """Return true only if each subset of the input item
               is not in confident_rules or less confident than item.
            """
            for item_old, conf_old in confident_rules:
                if set(item_old).issubset(set(item)) and 1.01 * conf_old >= conf:
                    return False
            return True

        for c in prior:
            for item, _ in sorted(self.frequent_set[c].items(), key=lambda x: (len(x[0]), -x[1])):
                conf = calculate_confidence(X, y, item, c)
                if better_interact(item, conf, self.confident_rule[c].items()):
                    self.confident_rule[c].push(item, conf)

    def generate_rule(self, features, prior):
        """Generate rules."""

        def rela_conf(conf, prior):
            return conf / (1.0001 - conf) * (1 - prior) / (prior + 0.0001)

        for c in self.confident_rule:
            for item, conf in sorted(self.confident_rule[c].items(), key=lambda x: len(x[0])):
                antecedents = tuple(map(lambda x: (features[x[0]], x[1]), item))
                rconf = rela_conf(conf, prior[c])
                self.rules.push((antecedents, c), rconf)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for (antecedents, _), _ in self.rules.items():
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)

    def fit(self, X, y, miss_val=[]):
        """Get the frequent itemsets by modified Apriori algorithm.

        Args:
            X: input data set, an n*p DataFrame.
            y: labels, an n*1 np.array.
            miss_val: a list of missing values.
        """
        # process the input
        features = list(X.columns)
        X = X.values
        classes = np.unique(y)

        if self.sample_size:
            indices = np.random.randint(len(X), size=self.sample_size)
        else:
            indices = np.arange(len(X))
        X_sample = X[indices]
        y_sample = y[indices]

        # calculate prior possibilities
        prior = {c: sum(y == c) / len(y) for c in classes}

        # init
        self.frequent_set = {c: MinHeap(self.n_freq) for c in classes}
        self.confident_rule = {c: MinHeap(self.n_conf) for c in classes}
        self.rules = MinHeap(self.n_conf)
        self.new_features = []

        # count 1-itemsets
        self.count_1_itemsets(X_sample, y_sample, miss_val)

        # select 1-order frequent sets that are most frequent
        self.select_frequent_1_itemsets()

        # count 2-itemsets
        self.count_2_itemsets(X_sample, y_sample)

        # select 1-itemsets and 2-itemsets that are most frequent
        self.select_frequent_itemsets()

        # select confident rules
        self.select_confident_rule(X, y, prior)

        # generate feature names
        self.generate_rule(features, prior)
        self.generate_feature_name()

    def transform(self, X):
        """Return the DataFrame of associated features.

        Args:
            X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated features.
        """

        def set_value(x, item):
            for index, value in item:
                if x[index] != value:
                    return 0
            return 1

        res = {}
        features = list(X.columns)
        indices = {f: i for i, f in enumerate(features)}
        for (antecedents, _), _ in self.rules.items():
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            item = [(indices[f], val) for f, val in antecedents]
            if feature_name not in res:
                res[feature_name] = np.apply_along_axis(set_value, 1, X.values, item)
        return pd.DataFrame(res, index=X.index)

    def transform_inter(self, X):
        """Return the DataFrame of interactive features.

        Args:
           X: the design matrix, a pd.DataFrame.
        Returns:
            a DataFrame contains generated interactive effects.
        """

        def set_value(x, item):
            for index, value in item:
                if x[index] != value:
                    return 0
            return 1

        res = {}
        features = list(X.columns)
        indices = {f: i for i, f in enumerate(features)}
        for (antecedents, _), _ in self.rules.items():
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            item = [(indices[f], val) for f, val in antecedents]
            if feature_name not in res and len(item) > 1:
                res[feature_name] = np.apply_along_axis(set_value, 1, X.values, item)
        return pd.DataFrame(res, index=X.index)


class ARAF_ctn(ARAF):
    def __init__(self, n_freq, n_conf, sample_size=None):
        super().__init__(n_freq, n_conf, sample_size)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for (antecedents, _), _ in self.rules.items():
            if len(antecedents) > 1:
                feature_name = '*'.join([str(f) for f, _ in antecedents])
                if feature_name not in self.new_features:
                    self.new_features.append(feature_name)

    def transform(self, X):
        res = {}
        for (antecedents, _), _ in self.rules.items():
            if len(antecedents) == 1:
                f1 = str(antecedents[0][0])[:-5]
                if f1 not in res:
                    res[f1] = X[f1].values
            else:
                f1, f2 = [str(f)[:-5] for f, _ in antecedents]
                feature_name = f1 + '*' + f2
                if feature_name not in res:
                    res[feature_name] = X[f1].values * X[f2].values
        return pd.DataFrame(res, index=X.index)

    def transform_inter(self, X):
        res = {}
        for (antecedents, _), _ in self.rules.items():
            if len(antecedents) > 1:
                f1, f2 = [str(f)[:-5] for f, _ in antecedents]
                feature_name = f1 + '*' + f2
                if feature_name not in res:
                    res[feature_name] = X[f1].values * X[f2].values
        return pd.DataFrame(res, index=X.index)
