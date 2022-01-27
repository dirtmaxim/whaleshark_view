import os
import random
import json
import numpy as np
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold, train_test_split
from configparser import ConfigParser
from transformations import transform_train, transform_test
from generator import DataGenerator

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def augmentations_example(folds_model, images, labels, sample_size=8):
    for fold_id, (train_index, test_index) in enumerate(folds_model.split(images, labels)):
        train_images, test_images = images[train_index], images[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                              test_size=0.1, random_state=seed)
        data_generator = DataGenerator(images=train_images, labels=train_labels, transform=transform_train,
                                       batch_size=sample_size, shape=shape)
        images, labels = data_generator[0]
        fig, axarr = plt.subplots(sample_size // 2, 2, figsize=(10, 10))

        for i in range(sample_size // 2):
            axarr[i][0].axis("off")
            axarr[i][1].axis("off")
            axarr[i][0].imshow((images[i][:, :, ::-1] * 255).astype(np.uint8))
            axarr[i][1].imshow((images[sample_size // 2 + i][:, :, ::-1] * 255).astype(np.uint8))

        plt.savefig("augmentations.png")
        break


def save_plots(history, fold_id):
    plt.clf()
    plt.plot(history.history["accuracy"], "b-")
    plt.plot(history.history["val_accuracy"], "g-")
    plt.title("Metrics")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["train", "val"])
    plt.savefig("logs/fold_{0}_accuracy.png".format(fold_id))
    plt.clf()
    plt.plot(history.history["loss"], "b-")
    plt.plot(history.history["val_loss"], "g-")
    plt.title("Metrics")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "val"])
    plt.savefig("logs/fold_{0}_loss.png".format(fold_id))


if __name__ == "__main__":
    with open("config.cfg", "r") as file:
        config = ConfigParser()
        config.read_file(file)

    backbone = config["Parameters"]["backbone"]
    seed = config["Parameters"].getint("seed")
    shape = config["Parameters"].getint("shape")
    dataset = config["Parameters"]["dataset_path"]
    batch_size = config["Parameters"].getint("batch_size")
    epochs = config["Parameters"].getint("epochs")
    random.seed(seed)

    images = []
    labels = []
    walks = list(os.walk(dataset))[1:]

    for path, dirs, files in walks:
        annotations = []

        for file in files:
            if os.path.splitext(file)[1] == ".json":
                with open(path + os.sep + file) as json_file:
                    data = json.load(json_file)

                    for entry in data:
                        annotations.append(entry[0])

        for file in files:
            if os.path.splitext(file)[1] != ".json":
                images.append(path + os.sep + file)

                if file in annotations:
                    labels.append(1)
                else:
                    labels.append(0)

    images = np.array(images)
    labels = np.array(labels)

    # Create folds model.
    folds_model = KFold(n_splits=4)

    # Create augmentations example and save it as image file.
    augmentations_example(folds_model, images, labels)

    # Train 4 models on each fold.
    for fold_id, (train_index, test_index) in enumerate(folds_model.split(images, labels)):
        fold = "models/fold_{0}".format(fold_id)

        if not os.path.exists(fold):
            os.makedirs(fold)

        train_images, test_images = images[train_index], images[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels,
                                                                              test_size=0.1, random_state=seed)

        ModelBuilder, _ = Classifiers.get(backbone)
        base_model = ModelBuilder(input_shape=(shape, shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(fold + "/epoch_{epoch:02d}.h5", monitor="val_accuracy", mode="max",
                                               save_weights_only=True, save_best_only=True, verbose=1),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.4, patience=2, verbose=1,
                                                 mode="max", min_lr=0.000000001),
        ]
        train_generator = DataGenerator(images=train_images, labels=train_labels, transform=transform_train,
                                        batch_size=batch_size, shape=shape)
        val_generator = DataGenerator(images=val_images, labels=val_labels, transform=transform_test,
                                      batch_size=batch_size, shape=shape, shuffle=False)
        history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, callbacks=callbacks)
        save_plots(history, fold_id)
        model.load_weights("{0}/".format(fold) + sorted(os.listdir(fold))[-1])
        model.save_weights("{0}/best.h5".format(fold))
