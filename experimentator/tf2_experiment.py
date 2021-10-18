from dataclasses import dataclass
from enum import IntFlag
from functools import cached_property
import glob
import logging
import os

import tensorflow as tf
from tensorflow.python.client import timeline # pylint: disable=no-name-in-module, unused-import

from experimentator import BaseExperiment, ExperimentMode, Callback, SubsetType

def print_tensor(x, message=None):
    def print_function(x):
        if message is not None:
            tf.print(message)
        tf.print(x)
        return x
    return tf.keras.layers.Lambda(print_function)(x)

class TensorFlowModelWrapper(tf.keras.Model): # pylint: disable=abstract-method
    @tf.function
    def train_step(self, inputs):
        with tf.GradientTape() as tape:
            results = self(inputs, training=True)
        if tf.math.reduce_any(tf.math.is_nan(results["loss"])):
            pass #return results
        grads = tape.gradient(results["loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return results
    @tf.function
    def test_step(self, inputs):
        return self(inputs, training=False)
    def load_weights(self, *args, **kwargs):
        super().load_weights(*args, **kwargs).expect_partial()

class TensorflowExperiment(BaseExperiment):
    run_options = None  # overwritten by callbacks
    run_metadata = None # overwritten by callbacks
    weights_formated_filename = "{epoch:04d}_weights"
    weights_file = None
    # def __init__(self, *args, **kwargs):
    #     tf.keras.backend.clear_session()
    #     super().__init__(*args, **kwargs)
    #     print(f"clearing session for {self.grid_sample}")

    @cached_property
    def metrics(self):
        return {}

    @cached_property
    def outputs(self):
        return {}

    @cached_property
    def device(self):
        return tf.config.list_logical_devices()[-1].name

    @cached_property
    def optimizer(self):
        return self.cfg["optimizer"]

    def load_weights(self, filename="auto"):
        if filename == "auto" or os.path.isdir(filename):
            dirname = os.path.dirname(self.cfg["filename"]) if filename == "auto" else filename
            try:
                filename = sorted(glob.glob(os.path.join(dirname, "*.index")))[-1].replace(".index", "")
            except IndexError:
                logging.error(f"Impossible to load weights in '{dirname}'. Use the 'filename' argument.")
                return
        logging.info(f"loading '{filename}'")
        self.weights_file = filename

    def save_weights(self, filename):
        self.train_model.save_weights(filename)

    def get_learning_rate(self):
        return self.optimizer.lr.numpy()

    def set_learning_rate(self, learning_rate):
        self.optimizer.lr = learning_rate

    # @cached_property
    # def checkpoint(self):
    #     tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.train_model, iterator=iter(self.batch_generator))

    # @cached_property
    # def manager(self):
    #     checkpoint_dir = os.path.join()
    #     tf.train.CheckpointManager(self.checkpoint, )

    @cached_property
    def inputs_specs(self): #pylint: disable=method-hidden
        data = next(iter(self.batch_generator(subset=self.subsets['training'])))[1]
        return self.inputs_specs_from_batch(data)

    def init_inputs(self, inputs_specs):
        if "inputs" not in self.__dict__:
            self.inputs_specs = inputs_specs
            _ = self.inputs

    @cached_property
    def inputs(self):
        return {name: tf.keras.Input(type_spec=type_spec, name=name) for name, type_spec in self.inputs_specs.items()}

    @cached_property
    def chunk_processors(self):
        return [CP for CP in self.cfg["chunk_processors"] if CP is not None] # cannot be a dict in case a ChunkProcessor is used twice

    @cached_property
    def chunk(self):
        chunk = self.inputs.copy() # copies the dictionary, but not its values (passed by reference) to be used again in the model instanciation
        for chunk_processor in self.chunk_processors:
            with tf.name_scope(chunk_processor.__class__.__name__):
                try:
                    chunk_processor(chunk)
                except BaseException as e:
                    if not self.cfg.get('robust', False):
                        raise e
                    logging.warning(f"{chunk_processor} skipped because of the following error: {e}")
        return chunk

    @cached_property
    def train_model(self):
        model = TensorFlowModelWrapper(self.inputs, {"loss": self.chunk["loss"]})
        model.compile(optimizer=self.optimizer)
        if self.weights_file:
            model.load_weights(self.weights_file)
        return model

    @cached_property
    def eval_model(self):
        model = TensorFlowModelWrapper(self.inputs, {"loss": self.chunk["loss"], **self.metrics}) # outputs are removed to accelerate evaluation during training. A dedicated eval+infer model should be used to perform both.
        model.compile(optimizer=self.optimizer)
        if self.weights_file:
            model.load_weights(self.weights_file)
        return model

    @cached_property
    def infer_model(self):
        model = TensorFlowModelWrapper(self.inputs, self.outputs)
        model.compile()
        if self.weights_file:
            model.load_weights(self.weights_file)
        return model

    def select_data(self, data):
        return {k:v for k,v in data.items() if k in self.inputs}

    @staticmethod
    def inputs_specs_from_batch(data):
        return {name: tf.TensorSpec(dtype=array.dtype, shape=array.shape) for name, array in data.items()}

    def batch_train(self, data, mode=ExperimentMode.TRAIN):
        self.init_inputs(self.inputs_specs_from_batch(data))
        model = self.train_model if mode & ExperimentMode.TRAIN else self.eval_model
        return model.train_step(self.select_data(data))

    def batch_eval(self, data):
        self.init_inputs(self.inputs_specs_from_batch(data))
        return self.eval_model.test_step(self.select_data(data))

    def batch_infer(self, data):
        self.init_inputs(self.inputs_specs_from_batch(data))
        return self.infer_model.test_step(self.select_data(data))


@dataclass
class EpochExponentialMovingAverage(Callback):
    """ 'decay' is highly dependant on the dataset size, dataset homogenity and batch size.
        /!\  tf_decay = 1 - torch_decay
    """
    decay: float
    def init(self, exp):
        self.train_model = exp.train_model
    def on_epoch_begin(self, **_):
        self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)
    def on_batch_end(self, cycle_type, **_):
        if cycle_type & SubsetType.TRAIN:
            self.ema.apply(self.train_model.trainable_variables)
    def on_epoch_end(self, **_):
        for var in self.train_model.trainable_variables:
            var.assign(self.ema.average(var))
        self.ema = None


@dataclass
class ProfileCallback(Callback):
    """ Enables profiling process between `batch_start` and `batch_stop` batches.
        Investigate with `tensorboard --logdir logdir --port 8020 --host `myip` --load_fast=false`
    """
    batch_start: int = 1
    batch_stop: int = 4
    mode: (IntFlag, int) = ExperimentMode.ALL
    def on_batch_begin(self, mode, epoch, batch, **_):
        if self.batch_start == batch:
            print("="*100 + "\nStart profiling\n" + "="*100)
            tf.profiler.experimental.start("logdir")
    def on_batch_end(self, mode, batch, **_):
        if batch == self.batch_stop:
            tf.profiler.experimental.stop()
            print("="*100 + "\nStop profiling\n" + "="*100)
