import numpy as np
import pandas as pd
import math
from collections import Counter


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


class Discretizer(object):
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