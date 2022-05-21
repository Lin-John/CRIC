import math
import numpy as np
import pandas as pd
import time
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score

from loaddata import load_data_adult, load_data_hd, load_data_dccc
from preprocessing import preprocess_X, preprocess_y, discretize_X, discretize_y, ohe_transform, split_transform
from araf import ARAF
from rit import RIT
from ric import RIC

import warnings

warnings.filterwarnings("ignore")


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
    res = {}
    model.fit(X_train.values, y_train)
    if "log_loss" in criteria:
        pred_proba = model.predict_proba(X_test.values)
        res["log_loss"] = round(log_loss(y_test, pred_proba), 4)
    if "accuracy_score" in criteria:
        pred = model.predict(X_test.values)
        res["accuracy_score"] = round(accuracy_score(y_test, pred), 4)
    if "roc_auc_score" in criteria:
        pred = model.predict(X_test.values)
        res["roc_auc_score"] = round(roc_auc_score(y_test, pred), 4) if len(np.unique(y_test)) == 2 else -1
    if 'mean_squared_error' in criteria:
        pred = model.predict(X_test.values)
        res['mean_squared_error'] = round(mean_squared_error(y_test, pred), 4)
    return res


def test_uci(load_data, models, criteria):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target].values.ravel()

    k = int(math.sqrt(len(sparse_features) + len(dense_features)))
    n_freq, n_conf = 10 * k, 10 * k
    n_class = 5

    methods = ["BASE", "RIT", "ARAF", "RIC-2", "RIC-5"]
    res = {method: {model: {criterion: [] for criterion in criteria} for model in models} for method in methods}
    n_feature = {method: [] for method in methods}
    running_time = {method: [] for method in methods}
    sfolder = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X, y):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features, sparse_features)
        y_train, y_test = preprocess_y(y_train, y_test)

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, dense_features,
                                                                    sparse_features, n_class, y_train)

        # One-hot encoding
        X_train_base, X_test_base = ohe_transform(X_train, X_test, sparse_features)
        running_time['BASE'].append(-1)
        n_feature['BASE'].append(len(X_test_base.columns))
        print('BASE', 'n_features:{}'.format(n_feature['BASE'][-1]))
        for model in models:
            scores = score((X_train_base, X_test_base, y_train, y_test), models[model], criteria)
            for criterion in scores:
                res['BASE'][model][criterion].append(scores[criterion])
            print(model, ', '.join(['{}:{}'.format(criterion, scores[criterion]) for criterion in scores]))

        # add interactions
        for method in methods:
            if method == 'RIT':
                inter_model = RIT(n_freq=n_freq, n_conf=n_conf)
            elif method == 'ARAF':
                inter_model = ARAF(n_freq=n_freq, n_conf=n_conf)
            elif method == 'RIC-2':
                inter_model = RIC(n_freq=n_freq, n_conf=n_conf, max_order=2)
            elif method == 'RIC-5':
                inter_model = RIC(n_freq=n_freq, n_conf=n_conf, max_order=5)
            else:
                continue
            time_start = time.time()
            inter_model.fit(X_train_de[sparse_features_de], y_train)
            time_end = time.time()
            X_train_inter = pd.concat((X_train_base, inter_model.transform_inter(X_train_de[sparse_features_de])),
                                      axis=1)
            X_test_inter = pd.concat((X_test_base, inter_model.transform_inter(X_test_de[sparse_features_de])), axis=1)

            running_time[method].append(time_end - time_start)
            n_feature[method].append(len(X_test_inter.columns))
            print(method,
                  'running_time:{}'.format(running_time[method][-1]),
                  'n_features:{}'.format(n_feature[method][-1]),
                  )
            for model in models:
                scores = score((X_train_inter, X_test_inter, y_train, y_test), models[model], criteria)
                for criterion in scores:
                    res[method][model][criterion].append(scores[criterion])
                print(model, ', '.join(['{}:{}'.format(criterion, scores[criterion]) for criterion in scores]))

    return res, n_feature, running_time


if __name__ == "__main__":
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)

    models = {'lr': lr}
    data_loaders = {'adult': load_data_adult, 'dccc': load_data_dccc, 'ce': load_data_ce}
    criteria = ["log_loss", "accuracy_score", "roc_auc_score"]

    data = {}
    for dataset in data_loaders:
        print(dataset)
        data[dataset] = test_uci(data_loaders[dataset], models, criteria)
    # with open('./data.pickle', 'wb') as f:
    #     pickle.dump(data, f)