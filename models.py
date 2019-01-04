"""Models for novelty prediction."""

from bisect import bisect_left
import tensorflow.keras as keras
import numpy as np
from scipy.stats import percentileofscore
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
 

def calibrate_confidences(confidences, validation_confidences):
    return np.array(
        [percentileofscore(validation_confidences, conf) for conf in confidences])


def compute_auroc(valid_scores, ood_scores):
    valid_scores = sorted(valid_scores)
    ood_scores = sorted(ood_scores)
    n = len(valid_scores) * 1.0
    percentiles = [bisect_left(ood_scores, s) / n for s in valid_scores]
    return np.mean(percentiles)


def linear_model(num_labels):
    return keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])

def hidden_model(num_labels, hidden_nodes=128):
    return keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax)
    ])

def hendrycks_mnist_model(num_labels, hidden_nodes=128):
    return keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
        keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
        keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax)
    ])

def papernot_conv_model(num_labels):
    return keras.models.Sequential([
        keras.layers.Conv2D(64, (8, 8), strides=(2,2), activation=tf.nn.relu,
                         padding='same'),
        keras.layers.Conv2D(128, (6,6), strides=(2,2), activation=tf.nn.relu,
                         padding='valid'),
        keras.layers.Conv2D(128, (5,5), strides=(1,1), activation=tf.nn.relu,
                         padding='valid'),
        keras.layers.Flatten(name='after_flatten'),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
])

class SoftmaxDetector(object):

    def __init__(self, keras_model):
        """Initialize given keras model whose output is a 1D softmax layer."""
        self._model = keras_model

    def train(self, id_data, id_labels, num_epochs=3):
        """Train model given input in-distribution data and labels."""
        # Labels should be 0, 1, ..., softmax_layer_size-1.
        self._model.compile(
        	optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        self._model.fit(id_data, id_labels, epochs=num_epochs)

    def validate(self, id_valid_data, id_valid_labels):
        """Give the model a validation set for calibration."""
        softmax_predictions = self._model.predict(id_valid_data)
        self._valid_confidences = np.array(sorted(np.amax(softmax_predictions, axis=1)))
        assert self._valid_confidences.shape[0] == id_valid_data.shape[0]
        return self._model.evaluate(id_valid_data, id_valid_labels)[1]


    def predict(self, data):
        """Predict confidences for a set of inputs."""
        # TODO: Output a tuple of the prediction AND confidence.
        softmax_predictions = self._model.predict(data)
        confidences = np.array(sorted(np.amax(softmax_predictions, axis=1)))
        auroc = compute_auroc(self._valid_confidences, confidences)
        confidences = calibrate_confidences(confidences, self._valid_confidences)
        return confidences, auroc


class NearestNeighborDetector(object):

    def __init__(self, feature_func):
        self._feature_func = feature_func

    def train(self, id_data, id_labels, num_epochs=-1):
        del num_epochs  # Unused.
        del id_labels  # Unused.
        id_data = self._feature_func(id_data)
        self._nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(id_data)

    def validate(self, id_valid_data, id_valid_labels):
        del id_valid_labels  # Unused.
        id_valid_data = self._feature_func(id_valid_data)
        distances, _ = self._nbrs.kneighbors(id_valid_data)
        self._valid_confidences = np.array(sorted(np.array([-d[0] for d in distances])))
        # TODO: Implement accuracy.
        return -1.0

    def predict(self, data):
        data = self._feature_func(data)
        distances, _ = self._nbrs.kneighbors(data)
        confidences = np.array(sorted(np.array([-d[0] for d in distances])))
        auroc = compute_auroc(self._valid_confidences, confidences)
        confidences = calibrate_confidences(confidences, self._valid_confidences)
        return confidences, auroc
