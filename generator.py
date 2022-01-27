import numpy as np
import cv2
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, images, labels, transform, batch_size, shape, shuffle=True):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        temp_images = [self.images[i] for i in indexes]
        x = self.__data_generation(temp_images)
        y = np.array([self.labels[i] for i in indexes], dtype=np.float32)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, temp_images):
        x = np.zeros(shape=(len(temp_images), self.shape, self.shape, 3), dtype=np.float32)

        for i, image_path in enumerate(temp_images):
            image = cv2.imread(image_path)
            transformed = self.transform(image=image)
            x[i] = transformed["image"].astype(np.float32) / 255

        return x
