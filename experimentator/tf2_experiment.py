import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline # pylint: disable=no-name-in-module, unused-import
from .base_experiment import BaseExperiment, lazyproperty
from .callbacked_experiment import Callback


class TensorflowExperiment(BaseExperiment):
    run_options = None  # overwritten by callbacks
    run_metadata = None # overwritten by callbacks

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        tf.config.set_soft_device_placement(False)
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            device_index = self.get("worker_id", self.get("GPU", 0))
            if device_index is None:
                device_index = 0
            visible_devices = [physical_devices[device_index]] # one single visible device for now
            tf.config.set_visible_devices(visible_devices, device_type="GPU")
            for device in visible_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)
        tf.config.run_functions_eagerly(self.get("eager", False))
        self.logger.debug(str(self.device) + str([device.name for device in tf.config.get_visible_devices()]))

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

    def load_weights(self, filename):
        self.train_model.load_weights(filename)

    def save_weights(self, filename):
        print("pad with 0s")
        self.train_model.save_weights(filename)

    # @lazyproperty
    # def checkpoint(self):
    #     tf.train.Checkpoint(step=tf.Variable(1), optimizer=self.optimizer, net=self.train_model, iterator=iter(self.batch_generator))

    # @lazyproperty
    # def manager(self):
    #     checkpoint_dir = os.path.join()
    #     tf.train.CheckpointManager(self.checkpoint, )

    @lazyproperty
    def inputs(self):
        # extract first dataset item which is not None to compute placeholders dimensions and types
        data = next(self.dataset.query_item(k) for k in iter(self.dataset.keys) if self.dataset.query_item(k))
        inputs = {}
        skipped = []
        for tensor_name in data:
            if isinstance(data[tensor_name], (np.ndarray, np.int32, int, float)):
                with tf.device(self.device):
                    inputs["batch_{}".format(tensor_name)] = tf.keras.Input(
                        dtype=tf.dtypes.as_dtype(data[tensor_name].dtype),
                        shape=data[tensor_name].shape,
                        name="batch_{}".format(tensor_name)
                    )
            else:
                skipped.append(tensor_name)
        if skipped:
            self.logger.debug("Skipped inputs: " + ", ".join(["{}({})".format(name, data[name]) for name in skipped]))
        return inputs

    @lazyproperty
    def chunk_processors(self):
        with tf.device(self.device):
            return [CP for CP in self.cfg["chunk_processors"] if CP is not None] # cannot be a dict in case a ChunkProcessor is used twice

    @lazyproperty
    def chunk(self):
        chunk = self.inputs.copy() # copies the dictionary, but not its values (passed by reference) to be used again in the model instanciation
        for chunk_processor in self.chunk_processors:
            with tf.name_scope(chunk_processor.__class__.__name__):
                with tf.device(self.device):
                    chunk_processor(chunk)
        return chunk

    @lazyproperty
    def train_model(self):
        with tf.device(self.device):
            return tf.keras.Model(self.inputs, {"loss": self.chunk["loss"], **self.metrics})

    @lazyproperty
    def eval_model(self):
        with tf.device(self.device):
            return tf.keras.Model(self.inputs, {"loss": self.chunk["loss"], **self.metrics, **self.outputs})

    @lazyproperty
    def infer_model(self):
        with tf.device(self.device):
            return tf.keras.Model(self.inputs, self.outputs)

    @tf.function
    def _batch_train(self, inputs):
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                results = self.train_model(inputs, training=True)
            grads = tape.gradient(results["loss"], self.train_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.train_model.trainable_weights))
        return results

    @tf.function
    def _batch_eval(self, inputs):
        return self.eval_model(inputs, training=False)

    @tf.function
    def _batch_infer(self, inputs):
        return self.infer_model(inputs, training=False)

    def select_data(self, data):
        return {k:v for k,v in data.items() if k in self.inputs}

    def batch_train(self, data):
        _ = self.train_model # forces lazyproperties to be created as attributes before wrapping with tf.function decorator
        return self._batch_train(self.select_data(data))
    def batch_eval(self, data):
        _ = self.eval_model # forces lazyproperties to be created as attributes before wrapping with tf.function decorator
        return self._batch_eval(self.select_data(data))
    def batch_infer(self, data):
        _ = self.infer_model # forces lazyproperties to be created as attributes before wrapping with tf.function decorator
        return self._batch_infer(self.select_data(data))



class ProfileCallback(Callback):
    # def __init__(self, exp):
    #     self.writer = tf.summary.create_file_writer("logdir")
    #     tf.debugging.experimental.enable_dump_debug_info("logdir", tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
    def on_epoch_begin(self, epoch, **_):
        return
        if epoch == 2:
            tf.profiler.experimental.start("logdir")
        if epoch == 4:
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
