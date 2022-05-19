import pandas as pd
import numpy as np
import random
from bisect import bisect_left


class PriorityQueue(object):
    """Keep a certain number of elements in order.
    Attributes:
        items: a list in the form of [(key, value)].
            the elements are ordered by value in an ascending order.
        size: the maximum number of the elements.
    """

    def __init__(self, size, items=[]):
        """Init the class"""
        self.items = items.copy()
        self.size = size

    def __len__(self):
        """"Return the number of elements in the list"""
        return len(self.items)

    def insert(self, key, val):
        """Insert a new item in the list"""
        index = bisect_left([x[1] for x in self.items], val)
        self.items.insert(index, (key, val))
        if len(self.items) > self.size:
            self.items.pop(0)

    def pop(self):
        """Delete and return the element with largest value"""
        return self.items.pop()

    def size_down(self):
        """Shrink the size of the list"""
        self.size -= 1

    def copy(self):
        """Return a copy of the list"""
        return PriorityQueue(self.size, self.items)


class RIC(object):
    """Get main effects and interactive features.

    Attributes:
        max_depth: the maximum depth of a chain, an integer.
        min_size: the maximum size of a frequent itemset, an integer.
        n_chain: the number of chains, an integer.
        n_freq: the number of the frequent itemsets, an integer.
        n_conf: the number of the confident rules, an integer.
        binary: whether the features are binary,
            if true, 0 stands for occurrence and 0 for obsence, only the 1s are concerned,
            if false, different values will be treated the same.
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

    def __init__(self, n_freq, n_conf, max_order=2, max_depth=1000, min_size=2, n_chain=1000,
                 binary=False, random_state=2020):
        """Init class with features that need to be encoded"""
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_chain = n_chain
        self.n_freq = n_freq
        self.n_conf = n_conf
        self.max_order = max_order
        self.binary = binary
        self.frequent_set = None
        self.confident_rule = None
        self.rules = None
        self.new_features = None
        self.freqs = None
        random.seed(random_state)

    def generate_chains(self, X, y):
        """Generate chains for different classes
        Args:
            X: the design matrix, a 2-D numpy.array.
            y: the labels, an 1-D numpy.array.

        Returns:
            a dict in the form of {c: [chain,...]}
            where c is a label, an integer,
            chain a tuple in the form of (values, counts),
            values is a line in the design matrix, an np.array,
            counts is a list of integers, recording how many times the value occurs.
        """

        def generate_chain(X):
            """Generate a chain
            Args:
                X: the design matrix, a 2-D numpy.array.
            Returns:
                a tuple in the form of (values, counts),
            """
            value = X[random.randint(0, len(X) - 1)]
            if self.binary:
                count = value.copy()
            else:
                count = [1] * len(value)
            size = sum(count)
            i = 1
            while i < self.max_depth:
                # sample new an instance unless the maximum depth is reached
                # or the number of remained items is sufficiently small
                old_count = count.copy()
                new_value = X[random.randint(0, len(X) - 1)]
                for p in range(len(new_value)):
                    if count[p] == i:
                        if value[p] == new_value[p]:
                            count[p] += 1
                        else:
                            size -= 1
                if size < self.min_size:
                    count = old_count
                    break
                i += 1
            return value, count

        chains = {c: [] for c in np.unique(y)}
        for c in chains:
            x_c = X[y == c]
            for _ in range(self.n_chain):
                chains[c].append(generate_chain(x_c))
        return chains

    def calculate_frequency(self, item, c, chains):
        """Calculate the frequency of an item"""

        def max_depth(item, chain):
            """Return the maximum depth of an item in a chain"""
            return min([chain[1][i] if chain[0][i] == val else 0 for i, val in item])

        if item in self.freqs[c]:
            return self.freqs[c][item]

        K, M = 0, 0
        for chain in chains:
            k = max_depth(item, chain)
            K += k
            if k < max(chain[1]) or max(chain[1]) == 0:
                M += 1
        self.freqs[c][item] = K / (K + M)
        return K / (K + M)

    def select_frequent_subset(self, temp_set, c, chains):
        """Select the most frequent subsets of temp_set.
        Args:
            temp_set: a tuple of items.
            c: the label.
            chains: a list of (value, count)..
        """
        temp_set = set(temp_set)
        for x in temp_set:
            item = (x,)
            if item not in self.freqs[c]:
                freq = self.calculate_frequency(item, c, chains)
                self.frequent_set[c].insert(item, freq)

        max_order = sum([len(item[0]) == 1 and item[0][0] in temp_set for item in self.frequent_set[c].items])
        max_order = min(max_order, self.max_order)
        for i in range(2, max_order + 1):
            token_a = self.frequent_set[c].copy()
            while len(token_a) > 1:
                a = set(token_a.pop()[0])
                token_a.size_down()
                if len(a) == 1 and a.issubset(temp_set):
                    token_b = token_a.copy()
                    while len(token_b) > 1:
                        b = set(token_b.pop()[0])
                        token_b.size_down()
                        if len(b) == i - 1 and b.issubset(temp_set) and len(a & b) == 0:
                            item = tuple(sorted(list(a | b)))
                            if item not in self.freqs[c]:
                                freq = self.calculate_frequency(item, c, chains)
                                self.frequent_set[c].insert(item, freq)
                            else:
                                freq = self.calculate_frequency(item, c, chains)
                            token_a.insert(item, freq)
                            token_b.insert(item, freq)

    def select_frequent_set(self, chains):
        """Extract frequent itemsets from chains and calculate thier frequency.
           The results are stored in self.frequent_set.
        Args:
            chains: a list of (value, count).
        """
        for c in chains:
            for item, count in chains[c]:
                depth = max(count)
                max_freq_item = tuple((i, item[i]) for i in range(len(count)) if count[i] == depth)
                if depth and len(max_freq_item):
                    self.select_frequent_subset(max_freq_item, c, chains[c])

    def calculate_confidence(self, item, freq, c, chains, prior, margin):
        """Calculte the confidence of an item.
        Args:
            item: a tuple of (index, value).
            freq: the frequency of the item, a float.
            c: the label.
            chains: a list of (value, count).
            prior: the probability of each class,
                a dict in the form of {c: freq}.
            margin: the marginal probability of the item.
                a dict in the form of {item: prob}.
        Returns:
            the confidence of the item.
        """

        # calculate the marginal probability
        if item not in margin:
            margin[item] = sum([prior[_] * self.calculate_frequency(item, _, chains[_])
                                for _ in prior])
        conf = prior[c] * freq / margin[item]
        return conf

    def select_confident_rule(self, chains, prior):
        """Extract rules from frequent itemsets and calculate thier confidence.
           The results are stored in self.confident_rule.
        Args:
            chains: a list of (value, count).
            prior: the probability of each class,
                a dict in the form of {c: freq}.
        """

        def better_interact(item, conf, confident_rules):
            """Return true only if each subset of the input item
               is not in confident_rules or less confident than item.
            """
            for item_old, conf_old in confident_rules:
                if set(item_old).issubset(set(item)) and 1.01 * conf_old >= conf:
                    return False
            return True

        margin = {}
        for c in prior:
            for item, freq in self.frequent_set[c].items[::-1]:
                conf = self.calculate_confidence(item, freq, c, chains, prior, margin)
                if better_interact(item, conf, self.confident_rule[c].items):
                    self.confident_rule[c].insert(item, conf)

    def generate_rule(self, features, prior):
        """Generate rules."""

        def rela_conf(conf, prior):
            return conf / (1.0001 - conf) * (1 - prior) / (prior + 0.0001)

        for c in self.confident_rule:
            for item, conf in self.confident_rule[c].items:
                antecedents = tuple(map(lambda x: (features[x[0]], x[1]), item))
                consequence = c
                rconf = rela_conf(conf, prior[c])
                self.rules.insert((antecedents, consequence), rconf)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for (antecedents, _), _ in self.rules.items:
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            if feature_name not in self.new_features:
                self.new_features.append(feature_name)

    def fit(self, X, y, miss_val=[]):
        """Get the useful features.

        Args:
            X: input data set, a DataFrame.
            y: labels, an n*1 DataFrame.
        """
        # process the input
        features = list(X.columns)
        X = X.values
        classes = np.unique(y)

        # init
        self.frequent_set = {c: PriorityQueue(self.n_freq) for c in classes}
        self.confident_rule = {c: PriorityQueue(self.n_conf) for c in classes}
        self.rules = PriorityQueue(self.n_conf)
        self.new_features = []
        self.freqs = {c: {} for c in classes}

        # calculate prior possibilities
        prior = {c: sum(y == c) / len(y) for c in classes}

        # generate chains
        chains = self.generate_chains(X, y)

        # select frequent itemsets
        self.select_frequent_set(chains)

        # select confident rules
        self.select_confident_rule(chains, prior)

        # generate feature names
        self.generate_rule(features, prior)
        self.generate_feature_name()

        self.freqs = None

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
        for (antecedents, _), _ in self.rules.items:
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
        for (antecedents, _), _ in self.rules.items:
            feature_name = ','.join(["{}={}".format(f, val) for f, val in antecedents])
            item = [(indices[f], val) for f, val in antecedents]
            if feature_name not in res and len(item) > 1:
                res[feature_name] = np.apply_along_axis(set_value, 1, X.values, item)
        return pd.DataFrame(res, index=X.index)


class RIC_ctn(RIC):
    """Select interactions of continuous features."""
    def __init__(self, n_freq, n_conf, max_order=2, max_depth=1000, min_size=2, n_chain=1000,
                 binary=False, random_state=2020):
        super().__init__(n_freq, n_conf, max_order, max_depth, min_size, n_chain, binary, random_state)

    def generate_feature_name(self):
        """Generate feature names for the selected itemsets."""
        for (antecedents, _), _ in self.rules.items:
            if len(antecedents) > 1:
                feature_name = '*'.join([str(f)[:-5] for f, _ in antecedents])
                if feature_name not in self.new_features:
                    self.new_features.append(feature_name)

    def transform(self, X):
        res = {}
        for (antecedents, _), _ in self.rules.items:
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
        for (antecedents, _), _ in self.rules.items:
            if len(antecedents) > 1:
                fs = [str(f)[:-5] for f, _ in antecedents]
                feature_name = '*'.join(fs)
                if feature_name not in res:
                    vals = np.ones(len(X))
                    for f in fs:
                        vals *= X[f].values
                    res[feature_name] = vals
        return pd.DataFrame(res, index=X.index)
