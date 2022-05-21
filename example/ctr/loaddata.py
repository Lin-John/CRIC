"""
data_folder/dataset_folder/split_type_folder/part_folder/decay_folder
"""

import pickle
import numpy as np
import pandas as pd
from os.path import join


def repeat_generator(generator):
    def generate_generator(*args, **kw):
        def wrapper():
            return generator(*args, **kw)
        return wrapper
    return generate_generator


def load_feat_info(fp):
    with open(fp, "rb") as handle:
        feat_info = pickle.load(handle)
    return (feat_info['target'], feat_info['dense_feat'],
            feat_info['sparse_feat'], feat_info['nunique_feat'])


def load_basic_data(part_folder, target, dense_feat, sparse_feat):
    columns = target + dense_feat + sparse_feat
    origin = np.load(join(part_folder, "train.npy"))
    data = pd.DataFrame(origin, columns=columns)
    return data[dense_feat+sparse_feat], data[target].values.ravel()


def load_inter_data(part_folder, target, dense_feat, sparse_feat, decay):
    inter_folder = part_folder if decay == 1 else join(part_folder, "decay_{}".format(decay))
    inter_feat_path = join(inter_folder, "inter_feat_name.txt")
    with open(inter_feat_path, "r") as f:
        inter_feat = f.readline().strip().split(' ')
    columns = target + dense_feat + sparse_feat + inter_feat
    origin = np.load(join(part_folder, "train.npy"))
    inter = np.load(join(inter_folder, "interaction.npy"))
    data = pd.DataFrame(np.concatenate([origin, inter], axis=1), columns=columns)
    return (data[dense_feat+sparse_feat+inter_feat], data[target].values.ravel()), inter_feat


@repeat_generator
def load_generator(part_names, split_type_folder, target, dense_feat, sparse_feat, use_interaction, decay):
    for part_name in part_names:
        part_folder = join(split_type_folder, part_name)
        if use_interaction:
            yield load_inter_data(part_folder, target, dense_feat, sparse_feat, decay)
        else:
            yield load_basic_data(part_folder, target, dense_feat, sparse_feat)


def load_data(dataset, split_type_folder, train_valid_test_id, use_interaction=False, decay=1):
    assert dataset in ["avazu", "criteo"], "invalid data set!"
    feat_info_path = join(split_type_folder, "feat_info.pkl")
    target, dense_feat, sparse_feat, nunique_feat = load_feat_info(feat_info_path)
    train_id, valid_id, test_id = train_valid_test_id
    args = [split_type_folder, target, dense_feat, sparse_feat, use_interaction, decay]
    train_generator = load_generator(["part"+str(i) for i in train_id], *args)
    valid_generator = load_generator(["part"+str(i) for i in valid_id], *args)
    test_generator = load_generator(["part"+str(i) for i in test_id], *args)
    return (train_generator, valid_generator, test_generator,
            target, dense_feat, sparse_feat, nunique_feat)


def shuffle_batch(X, y, seed=None):
    if seed is not None:
        np.random.seed(seed)
    indices = np.random.permutation(len(X))
    return X.iloc[indices], y[indices]