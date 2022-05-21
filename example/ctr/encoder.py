import numpy as np
from collections import Counter


class MyLabelEncoder(object):
    """Label encode the features.
    All the values appearing less than a given threshold in training set will be labeled by 0.

    Attributes:
        thresh: the threshold of occurrences.
        count: a dict recording the occurrences of each value.
        encode: a dict whose key is the original value and val is the corresponding code.
        decode: a dict whose key is the code and value is the corresponding value.
    """

    def __init__(self, thresh=1):
        """Init class with the threshold."""
        self.thresh = thresh
        self.count = {}
        self.encode = {}
        self.decode = {}

    def fit(self, X):
        """Fit a transformer on the records."""
        count = Counter(X)
        for val in count:
            if val not in self.count:
                self.count[val] = count[val]
            else:
                self.count[val] += count[val]
            if self.count[val] >= self.thresh and val not in self.encode:
                label = len(self.encode) + 1
                self.encode[val] = label
                self.decode[label] = val

    def transform(self, X):
        """Encode the the encoded feature."""
        return np.vectorize(lambda x: self.encode.get(x, 0))(X)

    def fit_transform(self, X):
        """fit a transformer on the records and
        return the result after encoding.
        """
        self.fit(X)
        return self.transform(X)

    def reverse(self, i):
        """return the original value of the corresponding code."""
        return self.decode[i]

    def reset(self):
        """Reset the encoder."""
        self.count = {}
        self.encode = {}
        self.decode = {}