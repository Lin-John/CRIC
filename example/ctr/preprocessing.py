import os
from os.path import join
import math
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold


def csv_data_generator(fp, dataset="avazu", names=None, batch_size=1000000):
    assert dataset in ["avazu", "criteo"], "invalid data set!"
    i = 0
    while True:
        if dataset == "avazu":
            data_batch = pd.read_csv(fp, sep=',', skiprows=range(1, i*batch_size+1), nrows=batch_size)
        elif dataset == "criteo":
            data_batch = pd.read_csv(fp, sep='\t', skiprows=i*batch_size, nrows=batch_size, names=names)
        if len(data_batch):
            yield data_batch
        else:
            break
        i += 1


def load_csv(dataset, dataset_folder):
    assert dataset in ["avazu", "criteo"], "invalid data set!"
    if dataset == "avazu":
        fp = join(dataset_folder, "train")
        target = ["click"]
        names = list(pd.read_csv(fp, sep=',', nrows=0).columns)
        dense_features = []
        sparse_features = [f for f in names if f != "click"]
        data = csv_data_generator(fp, dataset)
        return data, target, dense_features, sparse_features
    elif dataset == "criteo":
        fp = join(dataset_folder, "train.txt")
        target = ['click']
        dense_features = ['I'+str(i) for i in range(1, 14)]
        sparse_features = ['C'+str(i) for i in range(1, 27)]
        names = target + dense_features + sparse_features
        data = csv_data_generator(fp, dataset, names)
        return data, target, dense_features, sparse_features


def process(data, lbe, sparse_features, dense_features):
    for f in sparse_features:
        data[f] = lbe[f].fit_transform(data[f]).astype('int32')
    for f in dense_features:
        data[f] = np.vectorize(lambda x: int(math.log(x)**2) if x>2 else x)(data[f])
    return data


def save_feat_info(feat_info, feat_info_path):
    with open(feat_info_path, "wb") as handle:
        pickle.dump(feat_info, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_data(data, fold_index, split_type_folder):
    for i, part_index in enumerate(fold_index):
        print("now part %d" % (i + 1))
        part_folder = join(split_type_folder, "part{}".format(i+1))
        if not os.path.exists(part_folder):
            os.makedirs(part_folder)
        train_name = "train.npy"
        data_tmp = data[part_index]
        data_path = join(part_folder, train_name)
        np.save(data_path, data_tmp)


def split_data_random(data, target, dataset_folder, seed):
    sf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(sf.split(data, data[target].values.ravel()))

    split_type_folder = join(dataset_folder, "random_split")
    if not os.path.exists(split_type_folder):
        os.makedirs(split_type_folder)

    fold_index = np.array([ind for _, ind in folds])
    index_path = join(split_type_folder, "fold_index.npy")
    np.save(index_path, fold_index)
    print("save indices done")

    save_data(data.values, fold_index, split_type_folder)
    print("save data done")


def generate_interaction(part_folder, file_name, columns, cric):
    inter_folder = part_folder if cric.decay == 1 else join(part_folder, "decay_{}".format(cric.decay))
    if not os.path.exists(inter_folder):
        os.makedirs(inter_folder)
    inter_feat_path = join(inter_folder, "inter_feat_name.txt")
    with open(inter_feat_path, "w") as f:
        f.write(' '.join(cric.new_features))

    X = pd.DataFrame(np.load(join(part_folder, file_name)),
                     columns=columns)
    interaction = cric.transform(X).values
    inter_path = join(inter_folder, "interaction.npy")
    np.save(inter_path, interaction)
    print("save interactions done")