import os
from os.path import join
from sklearn.metrics import log_loss, roc_auc_score
from tensorflow import keras

from deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names, build_input_features
from deepctr.layers import custom_objects

from loaddata import load_data
from trainmodel import build_model, combine_model, train_model, train_combined_model

if __name__ == "__main__":
    dataset = "criteo"  # criteo, avazu
    opt = "adam"  # adagrad, adagrad
    exp = "baseline"  # baseline, inter
    model_type = "DeepFM"  # xDeepFM, DeepFM, WDL, DCN
    data_folder = join("data", dataset, "random_split")
    exp_folder = join("experiments", dataset, "random_split", model_type)

    num_trials = 1
    epochs = 10
    batch_size = 8192
    learning_rate = 1e-3
    patience = 5
    learning_rate_ft = 1e-5
    patience_ft = 1
    emb_dim = 16
    epochs_skip_es = 0

    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)

    result_path = join(exp_folder, "{}_result.txt".format(exp))
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            test_performances = f.readlines()
    else:
        test_performances = []

    if opt == "adagrad":
        optimizer = keras.optimizers.Adagrad
    elif opt == "adam":
        optimizer = keras.optimizers.Adam
    else:
        raise ValueError("Invalid optimizer")

    train_valid_test_id = [list(range(3, 11)), [2], [1]]
    if exp == "baseline":
        (train_generator, valid_generator, test_generator,
         target, dense_features, sparse_features, nunique_feat) = load_data(dataset, data_folder, train_valid_test_id)

        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=nunique_feat[feat] + 1, embedding_dim=emb_dim)
                                  for feat in sparse_features] + \
                                 [DenseFeat(feat, 1, ) for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        for i in range(num_trials):
            if i < len(test_performances):
                continue

            print("Starting trial", i + 1)

            model_checkpoint_file = join(exp_folder,
                                         "{}_trial{}.h5".format(exp, i))

            # build or load the model
            if not os.path.exists(model_checkpoint_file):
                model = build_model(model_type, linear_feature_columns, dnn_feature_columns)
            else:
                model = keras.models.load_model(model_checkpoint_file, custom_objects)

            # train the model
            model.compile(optimizer(learning_rate),
                          "binary_crossentropy",
                          metrics=['binary_crossentropy'], )
            train_model(model, feature_names,
                        train_generator, valid_generator,
                        epochs_skip_es, patience,
                        epochs, batch_size,
                        model_checkpoint_file)

            # fine-tune the model
            model = keras.models.load_model(model_checkpoint_file, custom_objects)
            model.compile(optimizer(learning_rate_ft),
                          "binary_crossentropy",
                          metrics=['binary_crossentropy'], )
            train_model(model, feature_names,
                        train_generator, valid_generator,
                        epochs_skip_es, patience_ft,
                        epochs, batch_size,
                        model_checkpoint_file)

            # evaluate the model
            model = keras.models.load_model(model_checkpoint_file, custom_objects)
            input_test, y_test = next(test_generator())
            input_test = [input_test[name] for name in feature_names]
            pred_ans = model.predict(input_test, batch_size=batch_size)

            test_logloss = round(log_loss(y_test, pred_ans, eps=1e-7), 7)
            test_auc = round(roc_auc_score(y_test, pred_ans), 7)
            print("test LogLoss", test_logloss)
            print("test AUC", test_auc)

            test_performances.append("{},{}\n".format(test_logloss, test_auc))
            with open(result_path, "w") as f:
                f.writelines(test_performances)

    elif exp == "inter":
        (train_generator, valid_generator, test_generator,
         target, dense_features, sparse_features, nunique_feat) = \
            load_data(dataset, data_folder, train_valid_test_id, use_interaction=True)

        fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=nunique_feat[feat] + 1, embedding_dim=emb_dim)
                                  for feat in sparse_features] + \
                                 [DenseFeat(feat, 1, )
                                  for feat in dense_features]
        dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

        _, inter_features = next(train_generator())
        inter_fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=2, embedding_dim=emb_dim)
                                        for feat in inter_features]
        inter_dnn_feature_columns = inter_fixlen_feature_columns
        inter_linear_feature_columns = inter_fixlen_feature_columns
        inter_feature_names = get_feature_names(inter_linear_feature_columns + inter_dnn_feature_columns)

        for i in range(num_trials):
            if i < len(test_performances):
                continue

            print("Starting trial", i + 1)

            model_checkpoint_file = join(exp_folder,
                                         "{}_trial{}.h5".format(exp, i))

            # build or load the joint model
            if not os.path.exists(model_checkpoint_file):
                base_model_checkpoint_file = join(exp_folder,
                                                  "{}_trial{}.h5".format("baseline", i))
                base_model = keras.models.load_model(base_model_checkpoint_file, custom_objects)
                inter_model = build_model(model_type, inter_linear_feature_columns, inter_dnn_feature_columns)
                model = combine_model(base_model, inter_model)
            else:
                model = keras.models.load_model(model_checkpoint_file, custom_objects)

            # train joint model
            model.compile(optimizer(learning_rate),
                          "binary_crossentropy",
                          metrics=['binary_crossentropy'], )
            train_combined_model(model, feature_names, inter_feature_names,
                                 train_generator, valid_generator,
                                 epochs_skip_es, patience,
                                 epochs, batch_size,
                                 model_checkpoint_file)

            # fine-tune joint model with the last period
            model = keras.models.load_model(model_checkpoint_file, custom_objects)
            model.trainable = True
            model.compile(optimizer(learning_rate_ft),
                          "binary_crossentropy",
                          metrics=['binary_crossentropy'], )
            train_combined_model(model, feature_names, inter_feature_names,
                                 train_generator, valid_generator,
                                 epochs_skip_es, patience_ft,
                                 epochs, batch_size,
                                 model_checkpoint_file)

            # evaluate joint model
            model = keras.models.load_model(model_checkpoint_file, custom_objects)
            (input_test, y_test), _ = next(test_generator())
            base_input_test = [input_test[name] for name in feature_names]
            inter_input_test = [input_test[name] for name in inter_feature_names]
            model = keras.models.load_model(model_checkpoint_file, custom_objects)
            pred_ans = model.predict([base_input_test, inter_input_test], batch_size=batch_size)

            test_logloss = round(log_loss(y_test, pred_ans, eps=1e-7), 7)
            test_auc = round(roc_auc_score(y_test, pred_ans), 7)
            print("test LogLoss", test_logloss)
            print("test AUC", test_auc)

            test_performances.append("{},{}\n".format(test_logloss, test_auc))
            with open(result_path, "w") as f:
                f.writelines(test_performances)

    else:
        raise ValueError("Invalid experiment")