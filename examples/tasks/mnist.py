import tensorflow as tf
import numpy as np

from experimentator import ChunkProcessor
from experimentator.tf2_experiment import TensorflowExperiment

class MNISTExperiment(TensorflowExperiment):
    batch_inputs_names = ["batch_input", "batch_target"]



class SimpleBackbone(ChunkProcessor):
    def __init__(self, classes):
        self.network = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(4, kernel_size=3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(8, kernel_size=3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, kernel_size=3),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(classes)
        ])
    def __call__(self, chunk):
        chunk["batch_logits"] = self.network(chunk["batch_input"], training=True)

