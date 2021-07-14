import os
import glob
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline # pylint: disable=no-name-in-module, unused-import
from .base_experiment import BaseExperiment, lazyproperty, ExperimentMode
from .callbacked_experiment import Callback

from .utils import SubsetType

#os.environ['AUTOGRAPH_VERBOSITY'] = "5"
#tf.config.set_soft_device_placement(False)
try:
    for device in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(device, enable=True)
except RuntimeError:
    print("Physical devices already initialized. Impossible to set memory growth.", file=os.sys.stderr)

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
        grads = tape.gradient(results["loss"], self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return results
    @tf.function
    def test_step(self, inputs):
        return self(inputs, training=False)

class TensorflowExperiment(BaseExperiment):
    run_options = None  # overwritten by callbacks
    run_metadata = None # overwritten by callbacks
    weights_suffix = "_weights"
    # def __init__(self, *args, **kwargs):
    #     tf.keras.backend.clear_session()
    #     super().__init__(*args, **kwargs)
    #     print(f"clearing session for {self.grid_sample}")

    @lazyproperty
    def metrics(self):
        return {}

    @lazyproperty
    def outputs(self):
        return {}

    @lazyproperty
    def device(self):
        return tf.config.list_logical_devices()[-1].name

    @lazyproperty
    def optimizer(self):
        return self.cfg["optimizer"]

    def load_weights(self, filename="auto"):
        if filename == "auto":
            dirname = os.path.dirname(self.cfg["filename"])
            try:
                filename = sorted(glob.glob(os.path.join(dirname, f"*{self.weights_suffix}.index")))[-1].replace(".index", "")
            except IndexError:
                logging.error(f"Impossible to load weights in '{dirname}'. Use the 'filename' argument.")
                return
        logging.info(f"loading '{filename}'")
        self.train_model.load_weights(filename)

    def save_weights(self, filename):
        self.train_model.save_weights(filename)

    def get_learning_rate(self):
        return self.optimizer.lr.numpy()

    def set_learning_rate(self, learning_rate):
        self.optimizer.lr = learning_rate

    # @lazyproperty
    # def checkpoint(self):
    #     tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.train_model, iterator=iter(self.batch_generator))

    # @lazyproperty
    # def manager(self):
    #     checkpoint_dir = os.path.join()
    #     tf.train.CheckpointManager(self.checkpoint, )

    def __init_inputs(self, data):
        if "inputs" not in self.__dict__:
            self.__data = data
            _ = self.inputs
            del self.__data

    @lazyproperty
    def inputs(self):
        try:
            data = self.__data # temporary variable holding the dictionary of batched data
        except AttributeError:
            # TODO: self.dataset.keys.all() can be slow
            data = next(iter(self.batch_generator(dataset=self.dataset, keys=self.dataset.keys.all())))[1]
        inputs = {}
        skipped = []
        for tensor_name in data:
            if isinstance(data[tensor_name], (np.ndarray, np.int32, int, float)):
                inputs[tensor_name] = tf.keras.Input(
                    dtype=tf.dtypes.as_dtype(data[tensor_name].dtype),
                    shape=data[tensor_name].shape[1:], # removing batch dimension in tf.keras.Input
                    name=tensor_name
                )
            else:
                skipped.append(tensor_name)
        if skipped:
            self.logger.debug("Skipped inputs: " + ", ".join(["{}({})".format(name, data[name]) for name in skipped]))
        return inputs

    @lazyproperty
    def chunk_processors(self):
        return [CP for CP in self.cfg["chunk_processors"] if CP is not None] # cannot be a dict in case a ChunkProcessor is used twice

    @lazyproperty
    def chunk(self):
        chunk = self.inputs.copy() # copies the dictionary, but not its values (passed by reference) to be used again in the model instanciation
        for chunk_processor in self.chunk_processors:
            with tf.name_scope(chunk_processor.__class__.__name__):
                chunk_processor(chunk)
        return chunk

    @lazyproperty
    def train_model(self):
        model = TensorFlowModelWrapper(self.inputs, {"loss": self.chunk["loss"]})
        model.compile(optimizer=self.optimizer)
        return model

    @lazyproperty
    def eval_model(self):
        model = TensorFlowModelWrapper(self.inputs, {"loss": self.chunk["loss"], **self.metrics, **self.outputs})
        model.compile(optimizer=self.optimizer)
        return model

    @lazyproperty
    def infer_model(self):
        model = TensorFlowModelWrapper(self.inputs, self.outputs)
        model.compile()
        return model

    # === START ===
    # @tf.function
    # def _batch_train(self, inputs):
    #     with tf.GradientTape() as tape:
    #         results = self.train_model(inputs, training=True)
    #     grads = tape.gradient(results["loss"], self.train_model.trainable_weights)
    #     self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_weights))
    #     return results

    # def batch_train(self, data, mode=None):
    #     self.__init_inputs(data)
    #     _ = self.train_model # forces lazyproperties to be created as attributes before wrapping with tf.function decorator
    #     return self._batch_train(self.select_data(data))
    # === STOP ===

    # @staticmethod
    # @tf.function
    # def _train_step(inputs, model, optimizer):
    #     with tf.GradientTape() as tape:
    #         results = model(inputs, training=True)
    #     grads = tape.gradient(results["loss"], model.trainable_weights)
    #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
    #     return results

    # @staticmethod
    # @tf.function
    # def _eval_step(inputs, model):
    #     return model(inputs, training=False)

    def select_data(self, data):
        return {k:v for k,v in data.items() if k in self.inputs}

    def batch_train(self, data, mode=ExperimentMode.TRAIN):
        self.__init_inputs(data)
        model = self.train_model if mode & ExperimentMode.TRAIN else self.eval_model
        return model.train_step(self.select_data(data))

    def batch_eval(self, data):
        self.__init_inputs(data)
        return self.eval_model.test_step(self.select_data(data))

    def batch_infer(self, data):
        self.__init_inputs(data)
        return self.infer_model.test_step(self.select_data(data))



class EpochExponentialMovingAverage(Callback):
    """ 'decay' is highly dependant on the dataset size, dataset homogenity and batch size.
        /!\  tf_decay = 1 - torch_decay
    """
    def __init__(self, decay):
        self.decay = decay
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

class TensorFlowProfilerExperiment(TensorflowExperiment):
    def run_batch(self, subset, batch_id, *args, **kwargs):
        with tf.profiler.experimental.Trace(subset.name, step_num=batch_id, _r=1):
            result = super().run_batch(*args, subset=subset, batch_id=batch_id, **kwargs)
        return result

class ProfileCallback(Callback):
    def __init__(self, batch_start=1, batch_stop=2, mode=ExperimentMode.ALL):
        self.batch_start = batch_start
        self.batch_stop = batch_stop
        self.mode = mode
    # def __init__(self, exp):
    #     self.writer = tf.summary.create_file_writer("logdir")
    #     tf.debugging.experimental.enable_dump_debug_info("logdir", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    def on_batch_begin(self, mode, epoch, batch, **_):
        if self.batch_start <= batch < self.batch_stop and mode & self.mode:
            tf.profiler.experimental.start("logdir")
    def on_batch_end(self, mode, batch, **_):
        if self.batch_start <= batch < self.batch_stop and mode & self.mode:
            tf.profiler.experimental.stop()
    # def on_batch_begin(self, epoch, **_):
    #     if 2 <= epoch < 4:
    #         tf.summary.trace_on(graph=True, profiler=True)
    # def on_batch_end(self, epoch, batch, **_):
    #     if 2 <= epoch < 4:
    #         with self.writer.as_default():
    #             tf.summary.trace_export(name="on_batch_end_{}.{}".format(epoch, batch), step=0, profiler_outdir="logdir")
#     def freeze(self, filename, output_nodes_names=None):
#         """Freezes the state of a session into a pruned computation graph.
#         Creates a new computation graph where variable nodes are replaced by
#         constants taking their current value in the session. The new graph will be
#         pruned so subgraphs that are not necessary to compute the requested
#         outputs are removed.
#         @param session The TensorFlow session to be frozen.
#         @param keep_var_names A list of variable names that should not be frozen,
#                             or None to freeze all the variables in the graph.
#         @param output_names Names of the relevant graph outputs.
#         @param clear_devices
#         @return The frozen graph definition.
#         """
#         assert filename[-3:] == ".pb", "You must provide a *.pb file"
#         output_nodes_names = output_nodes_names or list(self.outputs.keys())
#         # Remove the device directives from the graph for better portability
#         graph_definition = self.graph.as_graph_def()
#         for node in graph_definition.node: # pylint: disable=no-member
#             node.device = ""
#         frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(self.session, graph_definition, output_nodes_names)
#         tf.train.write_graph(frozen_graph, os.path.dirname(filename), os.path.basename(filename), as_text=False)

#         # compute the number of flops
#         opt = tf.profiler.ProfileOptionBuilder.float_operation()
#         flops = tf.profiler.profile(self.graph, options=opt)
#         total_flops = flops.total_float_ops
#         print("frozen graph saved in '{}' with {} flops".format(filename, total_flops))




# class ProfileBatch(Callback):
#     precedence = 95 # almost last
#     def __init__(self, exp):
#         self.loggdir = exp.get("loggdir", "profiler_logs")
#         self.profiler = tf.profiler.Profiler(graph=exp.graph)
#         self.exp = exp
#         self.summary = tf.summary.FileWriter(self.loggdir + '/train', graph=exp.graph, session=exp.session)
#         self.target_batch = 10

#     def on_batch_begin(self, batch_id, **_):
#         if batch_id != self.target_batch:
#             return
#         #tf.summary.trace_on(graph=True, profiler=True)
#         self.exp.run_metadata = tf.RunMetadata()
#         self.exp.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # pylint: disable=no-member
#         # tf.profiler.experimental.start(
#         #     self.loggdir, tf.profiler.ProfilerOptions(host_tracer_level=2)
#         # )
#     def on_batch_end(self, batch_id, subset, **_):
#         key = "{}_{}_{}".format(self.exp["experiment_id"], subset, batch_id)
#         if batch_id != self.target_batch:
#             return

#         if False:
#             # with self.summary.as_default():
#             #     for metric in self.exp.metrics.keys():
#             #         tf.summary.scalar(metric, state[metric], step=batch_id)
#             #     tf.summary.trace_export(
#             #         name="func_trace",
#             #         step=batch_id,
#             #         profiler_outdir=self.loggdir
#             #     )
#             self.summary.add_run_metadata(self.exp.run_metadata, "run_metadata", batch_id)
#             # for metric_name in self.exp.metrics.keys():
#             #     tf.summary.scalar(metric_name, state[metric_name])
#             #     self.summary.add_summary()#, step=batch_id))

#             self.profiler.add_step(batch_id, self.exp.run_metadata)


#             #self.profiler.profile_operations(opts)

#             # opts = (tf.profiler.ProfileOptionBuilder()
#             #     .with_max_depth(10)
#             #     .with_min_micros(1000)
#             #     .select(['accelerator_micros'])
#             #     .with_stdout_output()
#             #     .build())
#             #     #.time_and_memory()
#             #     #.with_stdout_output()
#             #     #.build())

#             # Or you can generate a timeline:

#         opts = (tf.profiler.ProfileOptionBuilder(
#                 tf.profiler.ProfileOptionBuilder.time_and_memory())
#                 .with_step(batch_id)
#                 .with_timeline_output("{}_timeline.ctf".format(key)).build())
#         self.profiler.profile_graph(options=opts)

#         if True:
#             opts = (tf.profiler.ProfileOptionBuilder(
#                     tf.profiler.ProfileOptionBuilder.time_and_memory())
#                     .with_step(batch_id)
#                     .with_timeline_output("{}_timeline_python.ctf".format(key)).build())
#             self.profiler.profile_python(options=opts)
#             #tf.profiler.experimental.stop()
#         #
#         #
#         if True:
#             tl = timeline.Timeline(self.exp.run_metadata.step_stats) # pylint: disable=no-member
#             with open("tf_timeline_{}".format(key), 'w') as f:
#                 f.write(tl.generate_chrome_trace_format())

#         self.exp.run_metadata = None
#         self.exp.run_options = None
