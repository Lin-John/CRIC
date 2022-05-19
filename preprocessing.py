import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OrdinalEncoder, OneHotEncoder, LabelEncoder
from discretizer import Discretizer


def preprocess_X(X_train, X_test, dense_features, sparse_features):
    """standardize the dense features and ordinal encode the sparse features. """
    X_train_pp, X_test_pp = X_train.copy(), X_test.copy()

    # standardize the dense features
    if dense_features:
        scaler = StandardScaler()
        X_train_pp[dense_features] = scaler.fit_transform(X_train_pp[dense_features])
        X_test_pp[dense_features] = scaler.transform(X_test_pp[dense_features])
    if sparse_features:
        oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        X_train_pp[sparse_features] = oe.fit_transform(X_train_pp[sparse_features])
        X_test_pp[sparse_features] = oe.transform(X_test_pp[sparse_features])
    return X_train_pp, X_test_pp


def discretize_X(X_train, X_test, dense_features, n_class, n_candidate=10, y_train=None):
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
#     if n_class < 2:
#         return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
#     disc_features = [str(f) + "_disc" for f in dense_features]
#     kbd = KBinsDiscretizer(n_bins=n_class, strategy='uniform', encode='ordinal')  # uniform, kmeans, quantile
#     kbd.fit(X_train[dense_features])
#     X_train_disc = pd.DataFrame(kbd.transform(X_train[dense_features]), columns=disc_features, index=X_train.index)
#     X_test_disc = pd.DataFrame(kbd.transform(X_test[dense_features]), columns=disc_features, index=X_test.index)
#     return X_train_disc, X_test_disc

    if n_class < 2:
        return pd.DataFrame(index=X_train.index), pd.DataFrame(index=X_test.index)
    se = Discretizer(dense_features, n_class, n_candidate)
    se.fit(X_train, y_train)
    X_train_disc = se.transform(X_train)
    X_test_disc = se.transform(X_test)
    return X_train_disc, X_test_disc


def preprocess_y(y_train, y_test, type='sparse'):
    """label encode the target."""
    y_train_pp, y_test_pp = y_train.copy(), y_test.copy()

    if type == 'dense':
        scaler = StandardScaler()
        y_train_pp = scaler.fit_transform(y_train_pp)
        y_test_pp = scaler.transform(y_test_pp)
    elif not np.issubdtype(y_train_pp.dtype, np.number):
    # label encode the target
        lbe = LabelEncoder()
        y_train_pp = lbe.fit_transform(y_train_pp)
        y_test_pp = lbe.transform(y_test_pp)
    else:
    # convert the target to intergers
        y_train_pp = y_train_pp.astype('int')
        y_test_pp = y_test_pp.astype('int')
    return y_train_pp, y_test_pp


def discretize_y(y_train, y_test, n_class):
    y_train_disc, y_test_disc = np.zeros(len(y_train)), np.zeros(len(y_test))
    qs = np.percentile(y_train, [100*i/n_class for i in range(n_class)])
    for i, thres in enumerate(qs):
        y_train_disc = np.where(y_train>=thres, i+1, y_train_disc)
        y_test_disc = np.where(y_test>=thres, i+1, y_test_disc)
    return y_train_disc, y_test_disc


def split_transform(X_train, X_test, dense_features, sparse_features, n_class, y_train=None):
    """Split the dense features"""
    X_train_de, X_test_de, sparse_features_de = X_train.copy(), X_test.copy(), sparse_features.copy()
    if dense_features:
        sparse_train, sparse_test = discretize_X(X_train, X_test, dense_features, n_class, y_train=y_train)
        X_train_de = pd.concat((X_train_de, sparse_train), axis=1)
        X_test_de = pd.concat((X_test_de, sparse_test), axis=1)
        sparse_features_de += list(sparse_test.columns)
    return X_train_de, X_test_de, sparse_features_de


def ohe_transform(X_train, X_test, sparse_features):
    """"Onehot encode the sparse features."""
    X_train_ohe, X_test_ohe = X_train.drop(sparse_features, axis=1), X_test.drop(sparse_features, axis=1)
    if sparse_features:
        ohe = OneHotEncoder(handle_unknown='ignore')
        ohe.fit(X_train[sparse_features])
        ohe_features = list(ohe.get_feature_names_out(sparse_features))
        ohe_train = pd.DataFrame(
            ohe.transform(X_train[sparse_features]).toarray(),
            columns=ohe_features,
            index=X_train.index
        )
        ohe_test = pd.DataFrame(
            ohe.transform(X_test[sparse_features]).toarray(),
            columns=ohe_features,
            index=X_test.index
        )
        X_train_ohe = pd.concat((X_train_ohe, ohe_train), axis=1)
        X_test_ohe = pd.concat((X_test_ohe, ohe_test), axis=1)
    return X_train_ohe, X_test_ohe