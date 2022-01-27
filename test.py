import os
import random
from classification_models.tfkeras import Classifiers
import cv2
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from configparser import ConfigParser
from transformations import transform_test

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def tta(image, model):
    flipped = np.fliplr(image)
    predicted = model.predict(np.expand_dims(image, axis=0))[0][0]
    predicted_flipped = model.predict(np.expand_dims(flipped, axis=0))[0][0]

    return (predicted + predicted_flipped) / 2


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

    mean_accuracy = []

    for fold_id, (train_index, test_index) in enumerate(folds_model.split(images, labels)):
        fold = "models/fold_{0}".format(fold_id)
        test_images, test_labels = images[test_index], labels[test_index]
        ModelBuilder, _ = Classifiers.get(backbone)
        base_model = ModelBuilder(input_shape=(shape, shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        model.load_weights("{0}/best.h5".format(fold))
        results = []

        for i, (image_path, label) in tqdm(enumerate(zip(test_images, test_labels)), total=len(test_images)):
            original = cv2.imread(image_path)
            transformed = transform_test(image=original)
            image = transformed["image"].astype(np.float32) / 255
            predicted = tta(image, model)
            captioned = original.copy()

            if predicted >= 0.5:
                cv2.putText(captioned, "GOOD: {0:.2f}".format(predicted), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                            (0, 255, 0), 1)
            else:
                cv2.putText(captioned, "BAD: {0:.2f}".format(predicted), (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5,
                            (0, 0, 255), 1)

            captioned_path = "/".join(image_path.split("/")[:-3]) + os.sep + "view_predicted"

            if not os.path.exists(captioned_path):
                os.makedirs(captioned_path)

            cv2.imwrite(captioned_path + os.sep + image_path.split("/")[-1], captioned)
            results.append(predicted)

        results = np.array(results)
        results[results >= 0.5] = 1
        results[results < 0.5] = 0
        accuracy = accuracy_score(test_labels, results)
        mean_accuracy.append(accuracy)
        print("Fold {0} Accuracy: {1:.3f}".format(fold_id, accuracy))

    print("Accuracy: {0:.3f}".format(np.mean(mean_accuracy)))
