import tensorflow as tf
from mlworkflow import DictDataset
from experimentator.dataset import Subset, SubsetType
import examples.tasks.mnist
from experimentator.tf2_chunk_processors import CastFloat, ExpandDims, SoftmaxCrossEntropyLoss, OneHotEncode

experiment_type = [examples.tasks.mnist.MNISTExperiment]

train, val = tf.keras.datasets.mnist.load_data()
subsets = [
    Subset("training", SubsetType.TRAIN, DictDataset(dict(enumerate(map(lambda x: {"input": x[0], "target": x[1]}, zip(*train)))))),
    Subset("validation", SubsetType.EVAL, DictDataset(dict(enumerate(map(lambda x: {"input": x[0], "target": x[1]}, zip(*val)))))),
]

batch_size = 16

globals().update(locals()) # required to use 'tf' in lambdas
chunk_processors = [
    CastFloat(tensor_names=["batch_input"]),
    ExpandDims(tensor_names=["batch_input"]),
    examples.tasks.mnist.SimpleBackbone(classes=10),
    lambda chunk: chunk.update({"batch_predictions": tf.nn.sigmoid(chunk['batch_logits'])}),
    OneHotEncode(tensor_name="batch_target", num_classes=10),
    SoftmaxCrossEntropyLoss(), # defines 'loss'
]

learning_rate=0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
