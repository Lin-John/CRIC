import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from araf import CRIC
from encoder import LabelEncoder, OneHotEncoder, SplitEncoder


def preprocess_X(X_train, X_test, dense_features):
    """standardize the dense features.
    """
    X_train_pp, X_test_pp = X_train.copy(), X_test.copy()

    # standardize the dense features
    if dense_features:
        scaler = StandardScaler()
        X_train_pp[dense_features] = scaler.fit_transform(X_train_pp[dense_features])
        X_test_pp[dense_features] = scaler.transform(X_test_pp[dense_features])
    return X_train_pp, X_test_pp


def preprocess_y(y_train, y_test):
    """label encode the target.
    """
    y_train_pp, y_test_pp = y_train.copy(), y_test.copy()

    # label encode the target
    if not np.issubdtype(y_train_pp.dtype, np.number):
        lbe = LabelEncoder('y', scale=False)
        y_train_pp = lbe.fit_transform(pd.DataFrame({'y':y_train_pp})).values.ravel()
        y_test_pp = lbe.transform(pd.DataFrame({'y':y_test_pp})).values.ravel()
    # convert the target to intergers
    else:
        y_train_pp = y_train_pp.astype('int')
        y_test_pp = y_test_pp.astype('int')
    return y_train_pp, y_test_pp


def discretize_y(y, n_class):
    y_disc = np.zeros(len(y))
    qs = np.percentile(y, [100*i/n_class for i in range(n_class)])
    for i, thres in enumerate(qs):
        y_disc = np.where(y>=thres, i+1, y_disc)
    return y_disc


def discretize(X_train, y_train, X_test, dense_features, n_class, n_candidate):
    """Discretize the dense features.
    Args:
        X_train: a DataFrame.
        y_train: an 1-D np.array.
        X_test: a DataFrame.
        dense_features: a list of names of dense features.
        n_class: the number of class after splitting.
        n_candidate: the number of thresholds for each searching.
    Returns:
        a DataFrame consists of discretized features.
    """
    if n_class < 2:
        return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
    se = SplitEncoder(dense_features, n_class, n_candidate)
    se.fit(X_train, y_train)
    X_train_dis = se.transform(X_train)
    X_test_dis = se.transform(X_test)
    return X_train_dis, X_test_dis


def lbe_transform(X_train, X_test, sparse_features, scale=True):
    """Label encode the sparse features."""
    X_train_lbe, X_test_lbe = X_train.copy(), X_test.copy()
    if sparse_features:
        lbe = LabelEncoder(sparse_features, scale)
        X_train_lbe[sparse_features] = lbe.fit_transform(X_train)
        X_test_lbe[sparse_features] = lbe.transform(X_test)
    return X_train_lbe, X_test_lbe


def ohe_transform(X_train, X_test, sparse_features):
    """"Onehot encode the sparse features."""
    X_train_ohe, X_test_ohe = X_train.drop(sparse_features, axis=1), X_test.drop(sparse_features, axis=1)
    if sparse_features:
        ohe = OneHotEncoder(sparse_features)
        ohe.fit(X_train)
        X_train_ohe[ohe.new_features] = ohe.transform(X_train)
        X_test_ohe[ohe.new_features] = ohe.transform(X_test)
    return X_train_ohe, X_test_ohe


def split_transform(X_train, X_test, y_train, dense_features, sparse_features,
                         n_class, n_candidate):
    """Split the dense features"""
    X_train_de, X_test_de, sparse_features_de = X_train.copy(), X_test.copy(), sparse_features.copy()
    if dense_features:
        sparse_train, sparse_test = discretize(X_train, y_train, X_test,
                                                dense_features, n_class, n_candidate)
        X_train_de[sparse_train.columns] = sparse_train
        X_test_de[sparse_test.columns] = sparse_test
        sparse_features_de += list(sparse_test.columns)
    return X_train_de, X_test_de, sparse_features_de


def araf_transform(X_train, X_test, y_train, sparse_features, miss_val, n_freq, n_conf):
    """Add the features generated from association rules to design matrix."""
    araf = CRIC(n_freq, n_conf)
    araf.fit(X_train[sparse_features], y_train, miss_val)
    X_train_araf = pd.concat((X_train, araf.transform(X_train[sparse_features])), axis=1)
    X_test_araf = pd.concat((X_test, araf.transform(X_test[sparse_features])), axis=1)
    if sparse_features:
        lbe = LabelEncoder(sparse_features)
        X_train_araf[sparse_features] = lbe.fit_transform(X_train_araf)
        X_test_araf[sparse_features] = lbe.transform(X_test_araf)
    return X_train_araf, X_test_araf


def araf_inter_transform(X_train, X_test, y_train, sparse_features, miss_val, n_freq, n_conf):
    """Add the features generated from association rules to design matrix."""
    araf = CRIC(n_freq, n_conf)
    araf.fit(X_train[sparse_features], y_train, miss_val)
    X_train_araf = pd.concat((X_train, araf.transform_inter(X_train[sparse_features])), axis=1)
    X_test_araf = pd.concat((X_test, araf.transform_inter(X_test[sparse_features])), axis=1)
    if sparse_features:
        lbe = LabelEncoder(sparse_features)
        X_train_araf[sparse_features] = lbe.fit_transform(X_train_araf)
        X_test_araf[sparse_features] = lbe.transform(X_test_araf)
    return X_train_araf, X_test_araf
