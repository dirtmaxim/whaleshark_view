import os
import sys
import numpy as np
import cv2
import tensorflow as tf
from classification_models.tfkeras import Classifiers
from configparser import ConfigParser
from transformations import transform_test

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


class ViewClassifier:
    def __init__(self):
        with open("config.cfg", "r") as file:
            config = ConfigParser()
            config.read_file(file)

        self.shape = config["Parameters"].getint("shape")
        ModelBuilder, _ = Classifiers.get(config["Parameters"]["backbone"])
        base_model = ModelBuilder(input_shape=(self.shape, self.shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.f0_model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        self.f0_model.load_weights("models/fold_0/best.h5")
        ModelBuilder, _ = Classifiers.get(config["Parameters"]["backbone"])
        base_model = ModelBuilder(input_shape=(self.shape, self.shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.f1_model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        self.f1_model.load_weights("models/fold_1/best.h5")
        ModelBuilder, _ = Classifiers.get(config["Parameters"]["backbone"])
        base_model = ModelBuilder(input_shape=(self.shape, self.shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.f2_model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        self.f2_model.load_weights("models/fold_2/best.h5")
        ModelBuilder, _ = Classifiers.get(config["Parameters"]["backbone"])
        base_model = ModelBuilder(input_shape=(self.shape, self.shape, 3), weights="imagenet", include_top=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.3)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.f3_model = tf.keras.models.Model(inputs=[base_model.input], outputs=[output])
        self.f3_model.load_weights("models/fold_3/best.h5")

    def __tta(self, image, model):
        flipped = np.fliplr(image)
        predicted = model.predict(np.expand_dims(image, axis=0))[0][0]
        predicted_flipped = model.predict(np.expand_dims(flipped, axis=0))[0][0]

        return (predicted + predicted_flipped) / 2

    def predict(self, image):
        transformed = transform_test(image=image)
        image = transformed["image"].astype(np.float32) / 255
        predicted = []

        for model in [self.f0_model, self.f1_model, self.f2_model, self.f3_model]:
            predicted.append(self.__tta(image, model))

        predicted = np.mean(predicted)

        return predicted


# Usage: python inference.py path_to_image
if __name__ == "__main__":
    vc = ViewClassifier()
    image = cv2.imread(sys.argv[1])
    result = vc.predict(image)

    if result >= 0.5:
        print("GOOD: {0:.2f}".format(result))
    else:
        print("BAD: {0:.2f}".format(result))
