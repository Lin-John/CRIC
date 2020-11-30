import math
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from process import preprocess_X, preprocess_y, lbe_transform, ohe_transform, split_transform
from getparameters import score, get_n_class
from araf import CRIC
from loaddata import load_data_adult, load_data_hd, load_data_dccc, load_data_ce


def test_uci(load_data, model, criterion):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target].values.ravel()
    k = int(math.sqrt(len(sparse_features)+len(dense_features)))

    res_ori_l, res_ori_o, res_araf_l, res_araf_o = [], [], [], []
    n_feature = {'ori_l': [], 'ori_o': [], 'araf_l': [], 'araf_o': []}
    running_time = []
    sfolder = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in sfolder.split(X, y):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features)
        y_train, y_test = preprocess_y(y_train, y_test)

        # split training set into new training set and valid set
        X_train1, X_valid, y_train1, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42,
                                                                stratify=y_train)

        # get the number of classes after discretizing
        n_splits = range(2, 10)
        n_candidate = 10
        n_conf = 15 * k
        n_freq = 5 * k
        n_split_best = get_n_class(X_train1, X_valid, y_train1, y_valid, dense_features, sparse_features, miss_val,
                                   n_splits, n_candidate, n_freq, n_conf, model, criterion)

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train,
                                                                    dense_features, sparse_features,
                                                                    n_split_best, n_candidate)

        time_start = time.time()
        araf = CRIC(15*k, 5*k, max_size=2)
        araf.fit(X_train_de[sparse_features_de], y_train, miss_val)
        time_end = time.time()
        running_time.append(time_end - time_start)

        # LabelEncoded original data
        X_train_ori_l, X_test_ori_l = lbe_transform(X_train, X_test, sparse_features)
        res_ori_l.append(score((X_train_ori_l, X_test_ori_l, y_train, y_test), model, criterion))
        n_feature['ori_l'].append(len(X_test_ori_l.columns))
        print(criterion, "of ori_l", res_ori_l[-1], n_feature['ori_l'][-1])

        # OneHotEncoded encoded original data
        X_train_ori_o, X_test_ori_o = ohe_transform(X_train, X_test, sparse_features)
        res_ori_o.append(score((X_train_ori_o, X_test_ori_o, y_train, y_test), model, criterion))
        n_feature['ori_o'].append(len(X_test_ori_o.columns))
        print(criterion, "of ori_o", res_ori_o[-1], n_feature['ori_o'][-1])

        # ARAF-L
        X_train_araf_l = pd.concat((X_train_ori_l, araf.transform(X_train_de[sparse_features_de])), axis=1)
        X_test_araf_l = pd.concat((X_test_ori_l, araf.transform(X_test_de[sparse_features_de])), axis=1)
        res_araf_l.append(score((X_train_araf_l, X_test_araf_l, y_train, y_test), model, criterion))
        n_feature['araf_l'].append(len(X_test_araf_l.columns))
        print(criterion, "of araf_l", res_araf_l[-1], n_feature['araf_l'][-1])

        # ARAF-O
        X_train_araf_o = pd.concat((X_train_ori_o, araf.transform_inter(X_train_de[sparse_features_de])), axis=1)
        X_test_araf_o = pd.concat((X_test_ori_o, araf.transform_inter(X_test_de[sparse_features_de])), axis=1)
        res_araf_o.append(score((X_train_araf_o, X_test_araf_o, y_train, y_test), model, criterion))
        n_feature['araf_o'].append(len(X_test_araf_o.columns))
        print(criterion, "of araf_o", res_araf_o[-1], n_feature['araf_o'][-1])

    methods = ["ori_l", "ori_o", "araf_l", "araf_o"]
    results = [res_ori_l, res_ori_o, res_araf_l, res_araf_o]
    for method, result in zip(methods, results):
        mean = round(np.mean(result), 4)
        std = round(np.std(result), 4)
        print(criterion, "of", method, "{}+-{}".format(mean, std))
    return res_ori_l, res_ori_o, res_araf_l, res_araf_o, running_time, n_feature


if __name__ == "__main__":
    # define the model
    lr = LogisticRegression(C=1, penalty='l1', solver="liblinear", random_state=42, max_iter=1000)

    load_data = load_data_ce
    model = lr  # mlp, dnn, rf, lr
    criterion = "log_loss"  # "log_loss", "accuracy_score", "roc_auc_score"
    res_ori_l, res_ori_o, res_araf_l, res_araf_o, running_time, n_feature = test_uci(load_data, model, criterion)
    for method in ["ori_l", "ori_o", "araf_l", "araf_o"]:
        print('n_features of', method, sum(n_feature[method]) / 5)
    print(sum(running_time) / 5)
