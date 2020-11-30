import math
import random
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from araf import CRIC_ctn
from loaddata import load_data_ccu
from process import preprocess_X, preprocess_y, discretize_y, split_transform
from getparameters import score


def test_ccu(load_data, criterion):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target].values.ravel()
    y = y / np.std(y)
    y_de = discretize_y(y, 3)
    n_freq, n_conf = 2500, 1225
    n_candidate = 10
    n_split_best = 5

    res = {c/10: 0 for c in range(1, 11)}
    sfolder = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X):
        # split data into training set and test set
        X_train, y_train_de, y_train = X.iloc[train_index], y_de[train_index], y[train_index]
        X_test, y_test_de, y_test = X.iloc[test_index], y_de[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features)
        y_train_de, y_test_de = preprocess_y(y_train_de, y_test_de)
        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train_de,
                                                                    dense_features, sparse_features,
                                                                    n_split_best, n_candidate)

        araf = CRIC_ctn(n_freq, n_conf)
        araf.fit(X_train_de[sparse_features_de], y_train_de, miss_val)
        # print(araf.transform(X_train).columns)
        X_train_araf = pd.concat((X_train, araf.transform(X_train)), axis=1)
        X_test_araf = pd.concat((X_test, araf.transform(X_test)), axis=1)
        X_train_araf, X_test_araf = preprocess_X(X_train_araf, X_test_araf, list(X_train_araf.columns))
        for c in res:
            res[c] += score((X_train_araf, X_test_araf, y_train, y_test), Lasso(alpha=c), criterion)
            print(c, res[c])

    c_best = 0.1
    for c in res:
        if res[c_best] > res[c]:
            c_best = c
    print('c_best', c_best, res[c_best]/5)

    model = Lasso(alpha=c_best)
    res_ori, res_araf = [], []
    for _ in range(200):
        # randomly split the data
        indices = list(range(len(X)))
        random.shuffle(indices)
        train_index, test_index = indices[:2 * len(X) // 3], indices[2 * len(X) // 3:]

        # split data into training set and test set
        X_train, y_train_de, y_train = X.iloc[train_index], y_de[train_index], y[train_index]
        X_test, y_test_de, y_test = X.iloc[test_index], y_de[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features)
        y_train_de, y_test_de = preprocess_y(y_train_de, y_test_de)

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train_de,
                                                                    dense_features, sparse_features,
                                                                    n_split_best, n_candidate)

        # original data
        res_ori.append(score((X_train, X_test, y_train, y_test), model, criterion))
        print(_, criterion, "of LabelEncoded origin", res_ori[-1], len(X_test.columns))

        # # ARAF
        araf = CRIC_ctn(n_freq, n_conf, max_size=5)
        araf.fit(X_train_de[sparse_features_de], y_train_de, miss_val)
        # print(araf.transform(X_train).columns)
        X_train_araf = pd.concat((X_train, araf.transform_inter(X_train)), axis=1)
        X_test_araf = pd.concat((X_test, araf.transform_inter(X_test)), axis=1)
        X_train_araf, X_test_araf = preprocess_X(X_train_araf, X_test_araf, list(X_train_araf.columns))
        res_araf.append(score((X_train_araf, X_test_araf, y_train, y_test), model, criterion))
        print(_, criterion, "of araf", res_araf[-1], len(X_test_araf.columns))

    methods = ["original", "ARAF"]
    results = [res_ori, res_araf]
    for method, result in zip(methods, results):
        mean = round(np.mean(result), 4)
        std = round(np.std(result), 4)
        print(criterion, "of", method, "{}+-{}".format(mean, std))
    return res_ori, res_araf


if __name__ == "__main__":
    load_data = load_data_ccu
    criterion = "mean_squared_error"  # "log_loss", "accuracy_score", "roc_auc_score"
    res_ori, res_araf = test_ccu(load_data, criterion)
    print(np.mean(res_araf), np.std(res_araf) / math.sqrt(200))
    print(np.mean(res_ori), np.std(res_ori) / math.sqrt(200))
