import numpy as np
import pandas as pd
from os.path import join
from preprocessing import load_csv, process, save_feat_info, split_data_random
from encoder import MyLabelEncoder


if __name__ == "__main__":
    seed = 2018
    dataset = "criteo"
    dataset_folder = join("./data", dataset)
    fp_clean = join(dataset_folder, "train_clean.txt")
    if dataset == "criteo":
        k = 10
    elif dataset == "avazu":
        k = 5

    data, target, dense_features, sparse_features = load_csv(dataset, dataset_folder)
    names = target + dense_features + sparse_features
    with open(fp_clean, "w") as f:
        f.write(','.join(names) + '\n')
    lbe = {f: xLabelEncoder(k) for f in sparse_features}
    for data_batch in data:
        if dataset == "criteo":
            data_batch[sparse_features] = data_batch[sparse_features].fillna('Missing', )
            data_batch[dense_features] = data_batch[dense_features].fillna(-1, )
            data_batch[target] = data_batch[target].fillna(0, )
        data_batch = process(data_batch, lbe, sparse_features, dense_features)
        data_batch.to_csv(fp_clean, columns=names, mode='a+', index=False, header=False, )
    nunique_feat = {f: len(lbe[f].encode) for f in lbe}
    feat_info = {"target": target,
                 "dense_feat": dense_features,
                 "sparse_feat": sparse_features,
                 "nunique_feat": nunique_feat}
    feat_info_path = join(dataset_folder, "feat_info.pkl")
    save_feat_info(feat_info, feat_info_path)

    data_clean = pd.read_csv(fp_clean, dtype=np.int32)

    split_data_random(data_clean, target, dataset_folder, seed)
    feat_info_path_1 = join(dataset_folder, "random_split", "feat_info.pkl")
    save_feat_info(feat_info, feat_info_path_1)