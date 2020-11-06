import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from encoder import LabelEncoder, OneHotEncoder, SplitEncoder
from araf import ARAF


def preprocess(X_train, X_test, y_train, y_test, dense_features, target):
    """Preprocess the data, including standardize the dense features and
       label encode the target.
    """
    X_train_pp, X_test_pp = X_train.copy(), X_test.copy()
    y_train_pp, y_test_pp = y_train.copy(), y_test.copy()

    # standardize the dense features
    if dense_features:
        scaler = StandardScaler()
        X_train_pp[dense_features] = scaler.fit_transform(X_train_pp[dense_features])
        X_test_pp[dense_features] = scaler.transform(X_test_pp[dense_features])

    # label encode the target
    if not np.issubdtype(y_train_pp.values.dtype, np.number):
        lbe = LabelEncoder(target, scale=False)
        y_train_pp[target] = lbe.fit_transform(y_train_pp)
        y_test_pp[target] = lbe.transform(y_test_pp)
    # convert the target to intergers
    else:
        y_train_pp[target] = y_train_pp[target].values.astype('int')
        y_test_pp[target] = y_test_pp[target].values.astype('int')
    return X_train_pp, X_test_pp, y_train_pp, y_test_pp


def discretize(X_train, y_train, X_test, dense_features, n_class, n_candidate):
    """Discretize the dense features.
    Args:
        X_train: a DataFrame.
        y_train: a DataFrame.
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
    araf = ARAF(n_freq, n_conf)
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
    araf = ARAF(n_freq, n_conf)
    araf.fit(X_train[sparse_features], y_train, miss_val)
    X_train_araf = pd.concat((X_train, araf.transform_inter(X_train[sparse_features])), axis=1)
    X_test_araf = pd.concat((X_test, araf.transform_inter(X_test[sparse_features])), axis=1)
    if sparse_features:
        lbe = LabelEncoder(sparse_features)
        X_train_araf[sparse_features] = lbe.fit_transform(X_train_araf)
        X_test_araf[sparse_features] = lbe.transform(X_test_araf)
    return X_train_araf, X_test_araf
