from os.path import join
from ric import RIC
from loaddata import load_data
from preprocessing import generate_interaction


if __name__ == "__main__":
    dataset = "criteo"
    split_type_folder = join("./data", dataset, "random_split")
    train_fns = [join("part{}".format(i), "train.npy") for i in range(3, 11)]
    train_valid_test_id = [list(range(3, 11)), [], []]
    train_generator, _, _, target, dense_features, sparse_features, _ = \
        load_data(dataset, split_type_folder, train_valid_test_id)
    columns = target + dense_features + sparse_features
    ric = RIC(200, 100, max_order=5, n_chain=1000, positive_class=True)
    for X_train, y_train in train_generator():
        ric.fit(X_train[sparse_features], y_train, [0])
    print("number of interactive features:", len(ric.new_features))
    ric.save(join(split_type_folder, r"ric.pkl"))

    for i in range(10):
        print("now part %d" % (i + 1))
        part_folder = join(split_type_folder, "part{}".format(i + 1))
        generate_interaction(part_folder, "train.npy", columns, ric)