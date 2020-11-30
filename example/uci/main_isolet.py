import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from araf import ARAF_ctn
from loaddata import load_data_isolet
from process import preprocess_X, preprocess_y, split_transform
from getparameters import score


def test_isolet(load_data, criterion):
    # load data
    df, dense_features, sparse_features, target, miss_val, task = load_data()
    X, y = df[sparse_features + dense_features], df[target].values.ravel()
    n_freq, n_conf = 2500, 1225
    n_candidate = 10
    n_split_best = 5

    res = {c / 10: 0 for c in range(1, 11)}
    sfolder = StratifiedKFold(n_splits=5, shuffle=True)
    for train_index, test_index in sfolder.split(X, y):
        # split data into training set and test set
        X_train, y_train = X.iloc[train_index], y[train_index]
        X_test, y_test = X.iloc[test_index], y[test_index]
        X_train, X_test = preprocess_X(X_train, X_test, dense_features)
        y_train, y_test = preprocess_y(y_train, y_test)

        for c in res:
            model = LogisticRegression(C=c, penalty='l2', solver="lbfgs", multi_class='multinomial',
                                       random_state=42, max_iter=10000)
            res[c] += score((X_train, X_test, y_train, y_test), model, criterion)
            print(c, res[c])

    c_best = 0.1
    for c in res:
        if res[c_best] < res[c]:
            c_best = c
    print('c_best', c_best, res[c_best] / 5)

    model = LogisticRegression(C=c_best, penalty='l2', multi_class="multinomial", solver="lbfgs",
                               random_state=42, max_iter=10000)

    res_ori, res_araf = [], []
    for _ in range(200):
        print(_)
        # split data into training set and test set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        X_train, X_test = preprocess_X(X_train, X_test, dense_features)
        y_train, y_test = preprocess_y(y_train, y_test)

        # LabelEncoded original data
        res_ori.append(score((X_train, X_test, y_train, y_test), model, criterion))
        print(criterion, "of origin", res_ori[-1], len(X_test.columns))

        # discretize continuous features
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train,
                                                                    dense_features, sparse_features,
                                                                    n_split_best, n_candidate)
        araf = CRIC_ctn(n_freq, n_conf, max_size=5)
        araf.fit(X_train_de[sparse_features_de], y_train, miss_val)

        # ARAF
        # X_train_araf = araf.transform(X_train)
        # X_test_araf = araf.transform(X_test)
        X_train_araf = pd.concat((X_train, araf.transform_inter(X_train)), axis=1)
        X_test_araf = pd.concat((X_test, araf.transform_inter(X_test)), axis=1)
        X_train_araf, X_test_araf = preprocess_X(X_train_araf, X_test_araf, list(X_train_araf.columns))
        res_araf.append(score((X_train_araf, X_test_araf, y_train, y_test), model, criterion))
        print(criterion, "of cric", res_araf[-1], len(X_test_araf.columns))

    methods = ["origin", "CRIC"]
    results = [res_ori, res_araf]
    for method, result in zip(methods, results):
        mean = round(np.mean(result), 4)
        std = round(np.std(result), 4)
        print(criterion, "of", method, "{}+-{}".format(mean, std))
    return res_ori, res_araf


if __name__ == "__main__":
    load_data = load_data_isolet
    criterion = "accuracy_score"
    res_ori, res_araf = test_isolet(load_data, criterion)
