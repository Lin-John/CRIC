mport os
import gc
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from deepctr.models import xDeepFM, DeepFM, WDL, DCN, AutoInt

from loaddata import shuffle_batch


def build_model(model_type, linear_feature_columns, dnn_feature_columns):
    if model_type == "DeepFM":
        model = DeepFM(linear_feature_columns,
                       dnn_feature_columns,
                       task="binary",
                       dnn_hidden_units=[400, 400, 400],
                       )
    elif model_type == "xDeepFM":
        model = xDeepFM(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            dnn_hidden_units=[400, 400],
            cin_layer_size=[200, 200, 200],
        )
    elif model_type == "WDL":
        model = WDL(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            dnn_hidden_units=[1024, 512, 256],
        )
    elif model_type == "DCN":
        model = DCN(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            dnn_hidden_units=[1024, 1024],
            cross_num=6,
        )
    else:
        model = AutoInt(
            linear_feature_columns,
            dnn_feature_columns,
            task="binary",
            dnn_hidden_units=[400, 400],
            att_embedding_size=64
        )
    return model


def combine_model(base_model, inter_model):
    base_part = keras.models.clone_model(base_model)
    base_part.set_weights(base_model.get_weights())
    base_part._name = "base_model"
    for layer in base_part.layers:
        layer._name = "base_" + layer.name
    base_input = base_part.input
    base_output = base_part.get_layer(base_part.layers[-2].name).output
    base_part.trainable = False

    inter_part = keras.models.clone_model(inter_model)
    inter_part.set_weights(inter_model.get_weights())
    inter_part._name = "inter_model"
    for layer in inter_part.layers:
        layer._name = "inter_" + layer.name
    inter_input = inter_part.input
    inter_output = inter_part.get_layer(inter_part.layers[-2].name).output

    x = layers.add([base_output, inter_output])
    x = tf.sigmoid(x)
    final_output = tf.reshape(x, (-1, 1))
    model = keras.Model(
        inputs=[base_input, inter_input],
        outputs=final_output,
    )
    return model


def train_model(model, feature_names,
                train_generator, valid_generator,
                epochs_skip_es, patience,
                epochs, batch_size,
                model_checkpoint_file):
    record = model_checkpoint_file.replace(".h5", ".txt")
    if os.path.exists(record):
        with open(record, "r") as f:
            rec_epoch, rec_file_count, best_valid_loss = list(map(float, f.readline().strip().split(',')))
    else:
        rec_epoch, rec_file_count, best_valid_loss = 0, -1, float("inf")
    patience_counter = 0

    try:
        input_valid, y_valid = next(valid_generator())
        input_valid = [input_valid[name] for name in feature_names]
        has_valid_set = True
    except StopIteration:
        has_valid_set = False

    for epoch in range(int(rec_epoch), epochs):
        breakout = False
        for file_count, data_batch in enumerate(train_generator()):
            if (epoch == int(rec_epoch) and file_count <= rec_file_count):
                continue
            print("epoch", epoch, "filecount", file_count)

            X_batch, y_batch = shuffle_batch(*data_batch)
            if not has_valid_set:
                X_batch, input_valid, y_batch, y_valid = train_test_split(X_batch, y_batch, test_size=0.2, stratify=y_batch)
                input_valid = [input_valid[name] for name in feature_names]
            input_batch = [X_batch[name] for name in feature_names]
            del data_batch, X_batch
            gc.collect()

            model.fit(input_batch, y_batch,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      )

            if epoch < epochs_skip_es:
                continue
            print(model.metrics_names)
            valid_loss, _= model.evaluate(input_valid, y_valid, batch_size=batch_size)

            if valid_loss < best_valid_loss:
                model.save(model_checkpoint_file)
                print(
                    "[%d-%d] model saved!. Valid loss improved from %.4f to %.4f"
                    % (epoch, file_count, best_valid_loss, valid_loss)
                )
                best_valid_loss = valid_loss
                patience_counter = 0
                with open(record, "w") as f:
                    f.write("{},{},{}".format(epoch, file_count, valid_loss))
            else:
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    with open(record, "w") as f:
                        f.write("{},{},{}".format(0, 0, best_valid_loss))
                    break
                patience_counter += 1
            gc.collect()

        if breakout:
            break


def train_combined_model(model, base_feature_names, int_feature_names,
                         train_generator, valid_generator,
                         epochs_skip_es, patience,
                         epochs, batch_size,
                         model_checkpoint_file):

    record = model_checkpoint_file.replace(".h5", ".txt")
    if os.path.exists(record):
        with open(record, "r") as f:
            rec_epoch, rec_file_count, best_valid_loss = list(map(float, f.readline().strip().split(',')))
    else:
        rec_epoch, rec_file_count, best_valid_loss = 0, 0, float("inf")
    patience_counter = 0

    try:
        (input_valid, y_valid), _ = next(valid_generator())
        base_input_valid = [input_valid[name] for name in base_feature_names]
        int_input_valid = [input_valid[name] for name in int_feature_names]
        has_valid_set = True
    except StopIteration:
        has_valid_set = False

    for epoch in range(int(rec_epoch), epochs):
        breakout = False
        for file_count, (data_batch, _) in enumerate(train_generator()):
            if (epoch == int(rec_epoch) and file_count <= rec_file_count):
                continue
            print("epoch", epoch, "filecount", file_count)

            X_batch, y_batch = shuffle_batch(*data_batch)
            if not has_valid_set:
                X_batch, input_valid, y_batch, y_valid = train_test_split(X_batch, y_batch, test_size=0.2, stratify=y_batch)
                base_input_valid = [input_valid[name] for name in base_feature_names]
                int_input_valid = [input_valid[name] for name in int_feature_names]
            base_input_batch = [X_batch[name] for name in base_feature_names]
            int_input_batch = [X_batch[name] for name in int_feature_names]
            del data_batch, X_batch
            gc.collect()

            model.fit([base_input_batch, int_input_batch], y_batch,
                      batch_size=batch_size,
                      epochs=1,
                      verbose=2,
                      )

            if epoch < epochs_skip_es:
                continue

            valid_loss, _ = model.evaluate([base_input_valid, int_input_valid], y_valid, batch_size=batch_size)

            if valid_loss < best_valid_loss:
                model.save(model_checkpoint_file)
                print(
                    "[%d-%d] model saved!. Valid loss improved from %.4f to %.4f"
                    % (epoch, file_count, best_valid_loss, valid_loss)
                )
                best_valid_loss = valid_loss
                patience_counter = 0
                with open(record, "w") as f:
                    f.write("{},{},{}".format(epoch, file_count, valid_loss))
            else:
                if patience_counter >= patience:
                    breakout = True
                    print("Early Stopping!")
                    with open(record, "w") as f:
                        f.write("{},{},{}".format(0, 0, best_valid_loss))
                    break
                patience_counter += 1
            gc.collect()

        if breakout:
            break