"""Models for novelty prediction."""

import keras
import numpy as np
from scipy.stats import percentileofscore
import tensorflow as tf
 

def calibrate_confidences(confidences, validation_confidences):
    return np.array(
        [percentileofscore(validation_confidences, conf) for conf in confidences])


def linear_model(num_labels):
    return keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax, name='out')
    ])

def hidden_model(num_labels, hidden_nodes=64):
    return keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(hidden_nodes, activation=tf.nn.relu),
        keras.layers.Dense(num_labels, activation=tf.nn.softmax)
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
        self._valid_confidences = np.amax(softmax_predictions, axis=1)
        assert self._valid_confidences.shape[0] == id_valid_data.shape[0]


    def predict(self, data):
        """Predict confidences for a set of inputs."""
        # TODO: Output a tuple of the prediction AND confidence.
        softmax_predictions = self._model.predict(data)
        confidences = np.amax(softmax_predictions, axis=1)
        confidences = calibrate_confidences(confidences, self._valid_confidences)
        return confidences
