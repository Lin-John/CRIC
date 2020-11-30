import numpy as np
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, mean_squared_error
from araf import CRIC
from process import lbe_transform, split_transform


def score(data, model, criterion):
    """Calculate the performance of the model on the dataset.
    Args:
        data: the dataset, a tuple in the form of (X_train, X_test, y_train, y_test),
              each element is a dataframe.
        model: the model, needs to have methods named "fit", "predict" and "predict_proba".
        criterion: a string, could be "log_loss", "accuracy_score" or "roc_auc_score".
    Returns:
        the logloss, accuracy or AUC on the test set for a model trained on the training set.
    """
    X_train, X_test, y_train, y_test = data
    model.fit(X_train, y_train)
    if criterion == "log_loss":
        pred_proba = model.predict_proba(X_test)
        res = round(log_loss(y_test, pred_proba), 4)
    elif criterion == "accuracy_score":
        pred = model.predict(X_test)
        res = round(accuracy_score(y_test, pred), 4)
    elif criterion == "roc_auc_score":
        pred = model.predict(X_test)
        res = round(roc_auc_score(y_test, pred), 4)
    elif criterion == 'mean_squared_error':
        pred = model.predict(X_test)
        res = round(mean_squared_error(y_test, pred), 4)
    return res


def get_n_freq_conf(X_train, X_test, y_train, y_test, sparse_features, miss_val,
                    n_freqs, n_confs, model, criterion, ret_scores=False):
    """Select the optimal number of frequent sets and confident rules."""
    classes = np.unique(y_train)
    scores = []
    for n_freq, n_conf in zip(n_freqs, n_confs):
        araf = CRIC(n_freq, n_conf)
        araf.fit(X_train[sparse_features], y_train, miss_val)

        X_train_araf, X_test_araf = X_train.copy(), X_test.copy()
        X_train_araf[araf.new_features] = araf.transform(X_train[sparse_features])[araf.new_features]
        X_test_araf[araf.new_features] = araf.transform(X_test[sparse_features])[araf.new_features]
        X_train_araf, X_test_araf = lbe_transform(X_train_araf, X_test_araf, sparse_features)

        scores.append(score((X_train_araf, X_test_araf, y_train, y_test), model, criterion))
        print(n_freq, n_conf, scores[-1])

    if ret_scores:
        return scores
    else:
        index = scores.index(min(scores)) if criterion == "log_loss" else scores.index(max(scores))
        return n_freqs[index], n_confs[index]


def grid_search(X_train, X_test, y_train, y_test, dense_features, sparse_features, miss_val,
                n_splits, n_candidate, n_freqs, n_confs, model, criterion):
    """calculate the criterion of different combinations of n_split and n_conf"""
    res = []
    for n_split in n_splits:
        print("n_split=", n_split)
        X_train_de, X_test_de, sparse_features_de = split_transform(X_train, X_test, y_train,
                                                                    dense_features, sparse_features,
                                                                    n_split, n_candidate)
        scores = get_n_freq_conf(X_train_de, X_test_de, y_train, y_test, sparse_features_de, miss_val,
                                 n_freqs, n_confs, model, criterion, ret_scores=True)
        res.append(scores)
    return res


def get_n_class(X_train, X_test, y_train, y_test, dense_features, sparse_features, miss_val,
                n_splits, n_candidate, n_freq, n_conf, model, criterion):
    """Select the optimal number of class after discretization."""
    if not dense_features:
        return 0

    scores = grid_search(X_train, X_test, y_train, y_test, dense_features, sparse_features, miss_val,
                         n_splits, n_candidate, [n_freq], [n_conf], model, criterion)
    n_split_best, n_conf_best, score_best = 0, 0, float('inf') if criterion == "log_loss" else -float('inf')
    for n_split, scores_split in zip(n_splits, scores):
        score_conf = scores_split[0]
        if ((criterion == "log_loss" and score_conf <= score_best) or
                (criterion != "log_loss" and score_conf >= score_best)):
            n_split_best, score_best = n_split, score_conf
    return n_split_best
