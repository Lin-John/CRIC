import math
import random
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.model_selection import KFold

from loaddata import load_data_ccu, load_data_isolet
from preprocessing import preprocess_X, preprocess_y, discretize_X, discretize_y, ohe_transform, split_transform
from araf import ARAF_ctn
from rit import RIT_ctn
from ric import RIC_ctn


def score(data, model, criteria):
    """Calculate the performance of the model on the dataset.
    Args:
        data: the dataset, a tuple in the form of (X_train, X_test, y_train, y_test),
              each element is a dataframe.
        model: the model, needs to have methods named "fit", "predict" and "predict_proba".
        criteria: a list of "log_loss", "accuracy_score" or "roc_auc_score".
    Returns:
        the logloss, accuracy or AUC on the test set for a model trained on the training set.
    """
    X_train, X_test, y_train, y_test = data
    model.fit(X_train.values, y_train)
    if "log_loss" in criteria:
        pred_proba = model.predict_proba(X_test.values)
        return round(log_loss(y_test, pred_proba), 4)
    if "accuracy_score" in criteria:
        pred = model.predict(X_test.values)
        return round(accuracy_score(y_test, pred), 4)
    if "roc_auc_score" in criteria:
        pred = model.predict(X_test.values)
        return round(roc_auc_score(y_test, pred), 4) if len(np.unique(y_test)) == 2 else -1
    if 'mean_squared_error' in criteria:
        pred = model.predict(X_test.values)
        return round(mean_squared_error(y_test, pred), 4)
    return


def test_isolet(load_data, criterion):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target].values.ravel()

    n_candidate = 10
    n_class = len(np.unique(y))
    n_split = 5
    n_freq, n_conf = 2500 // n_class, 1225

    res_c = {c/10:0 for c in range(1, 11)}
    sfolder = KFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features, sparse_features)

        for c in res_c:
            res_c[c] += score((X_train, X_test, y_train, y_test), Lasso(alpha=c), criterion)

    c_best = 0.3
    for c in res_c:
        if res_c[c_best] > res_c[c]:
            c_best = c
    print('c_best', c_best, res_c[c_best]/5)

    model = LogisticRegression(C=c_best, penalty='l2', multi_class="multinomial", solver="lbfgs",
                               random_state=42, max_iter=10000)

    methods = ["base", "rit", "araf", "ric"]
    res = {method: [] for method in methods}
    for _ in range(10):
        print(_)
        # randomly split the data
        indices = list(range(len(X)))
        random.shuffle(indices)
        train_index, test_index = indices[:2 * len(X) // 3], indices[2 * len(X) // 3:]

        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features, sparse_features)
        y_train_de, y_test_de = y_train, y_test

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, dense_features, sparse_features,
                                                                    n_split, y_train)

        # original features
        res['base'].append(score((X_train, X_test, y_train, y_test), model, criterion))
        print('base', res['base'][-1], len(X_test.columns))

        # add interactions
        for method in methods:
            if method == 'rit':
                inter_model = RIT_ctn(n_freq=n_class * n_freq, n_conf=n_conf, theta_0=0.3, theta_1=0.5, n_tree=20,
                                      num_splits=5)
            elif method == 'araf':
                inter_model = ARAF_ctn(n_freq=n_freq, n_conf=n_conf)
            elif method == 'ric-2':
                inter_model = RIC_ctn(n_freq=n_freq, n_conf=n_conf, n_chain=300, max_order=2)
            elif method == 'ric':
                inter_model = RIC_ctn(n_freq=n_freq, n_conf=n_conf, n_chain=300, max_order=5)
            else:
                continue
            inter_model.fit(X_train_de[sparse_features_de], y_train_de)
            X_train_inter = pd.concat((X_train, inter_model.transform_inter(X_train)), axis=1)
            X_test_inter = pd.concat((X_test, inter_model.transform_inter(X_test)), axis=1)
            X_train_inter, X_test_inter = preprocess_X(X_train_inter, X_test_inter, list(X_train_inter.columns), [])

            res[method].append(score((X_train_inter, X_test_inter, y_train, y_test), model, criterion))
            print(method, res[method][-1], len(X_test_inter.columns))

    return res


if __name__ == "__main__":
    load_data = load_data_isolet
    criterion = "accuracy_score"
    res = test_isolet(load_data, criterion)
