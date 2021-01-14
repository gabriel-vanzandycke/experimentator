import os
import random
import logging
import numpy as np
from .base_experiment import BaseExperiment, lazyproperty
from .utils import OutputInhibitor
if False: # pylint: disable=using-constant-test
    lazyproperty = property
import tensorflow as tf

class TensorflowExperiment(BaseExperiment):
    run_options = None  # overwritten by callbacks
    run_metadata = None # overwritten by callbacks

    def init_model(self, weights=None):
        # pylint: disable=pointless-statement
        tf.config.set_soft_device_placement(False)
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.set_visible_devices(physical_devices[int(self["GPU"])], device_type="GPU")
        tf.config.experimental.set_memory_growth(physical_devices[int(self["GPU"])], enable=True)
        tf.config.run_functions_eagerly(self.get("eager", False))
        print(tf.config.get_visible_devices())
        
        self.process # as a lazyproperty (no need to use the parenthisis)
        
        self.metrics
        random_seed = int(self["manager_id"].replace("_","")[4:])
        #tf.set_random_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        # if weights is None and "weights" in self.logger:
        #     try:
        #         weights = self.logger["weights"]
        #     except FileNotFoundError:
        #         print("Last weights have been deleted. Using random weights.")
        # if weights is not None:
        #     self.set_weights(weights)

    @lazyproperty
    def optimizer(self):
        with self.graph.device(self.get("device", "/gpu:0")): # pylint: disable=not-context-manager
            with tf.name_scope("optimizer"):
                optimizer = self["optimizer"]()
        return optimizer

    # set weights from variable
    def set_weights(self, weights):
        print("Loading weights...", end="")
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.session.run([k.assign(v) for k, v in zip(var_list, weights)])
        print(" Done.")

    # get weights from variable
    def get_weights(self):
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self.session.run(var_list)

    @property
    def weights(self):
        return 0#self.get_weights()

    @property
    def device(self):
        return "/gpu:{}".format(self["GPU"])

    @lazyproperty
    def chunk_processors(self):
        chunk_processors = {}
        for ChunkProcessor in self["chunk_processors"]:
            if ChunkProcessor is None:
                continue
            chunk_processor = ChunkProcessor()
            chunk_processors[chunk_processor.__class__.__name__] = chunk_processor
        return chunk_processors

    @lazyproperty # done only once in tensorflow: to create the graph
    def process(self):
        if True:#with tf.device(self.device):
            for name, chunk_processor in self.chunk_processors.items():
                with tf.name_scope(name):
                    chunk_processor(self.chunk)

    @lazyproperty
    def inputs(self):
        data = self.dataset.query_item(next(iter(self.dataset.keys)))
        with tf.device(self.device):
            inputs = {
                "flag_rotate": tf.keras.Input(shape=(1), name="flag_rotate"),
            }
            skipped = []
            for tensor_name in data:
                if isinstance(data[tensor_name], (np.ndarray, np.int32, int, float)):
                    inputs["batch_{}".format(tensor_name)] = tf.keras.Input(
                        dtype=tf.dtypes.as_dtype(data[tensor_name].dtype),
                        shape=data[tensor_name].shape,
                        name="batch_{}".format(tensor_name)
                    )
                else:
                    skipped.append(tensor_name)
            print("Skipped inputs: " + ", ".join(["{}({})".format(name, data[name]) for name in skipped]))
        #inputs["width"] = data["input_image"].shape[1] # doesn't work with model expecting tensors as input and not values
        #inputs["height"] = data["input_image"].shape[0] # doesn't work
        return inputs

    @lazyproperty
    def chunk(self):
        return self.inputs.copy() # copies the dictionary, but not its values (passed by reference)

    @lazyproperty
    def train_model(self):
        return tf.keras.Model(self.inputs, {"loss": self.chunk["loss"], **self.metrics})

    @lazyproperty
    def eval_model(self):
        return tf.keras.Model(self.inputs, {"loss": self.chunk["loss"], **self.metrics, **self.outputs})

    @lazyproperty
    def infer_model(self):
        return tf.keras.Model(self.inputs, self.outputs)

    @lazyproperty
    def optimizer(self):
        if True: #with tf.graph(self.device):
            with tf.name_scope("optimizer"):
                optimizer = self["optimizer"]()
        return optimizer


    @tf.function
    def _batch_train(self, inputs):
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

    @property
    def learning_rate(self):
        return 42

    def batch_train(self, data):
        inputs = {k:v for k,v in {"flag_rotate": np.array([0]*self.batch_size), **data}.items() if k in self.inputs}
        return self._batch_train(inputs)
    def batch_eval(self, data):
        inputs = {k:v for k,v in {"flag_rotate": np.array([0]*self.batch_size), **data}.items() if k in self.inputs}
        return self._batch_eval(inputs)
    def batch_infer(self, data):
        inputs = {k:v for k,v in {"flag_rotate": np.array([0]*self.batch_size), **data}.items() if k in self.inputs}
        return self._batch_infer(inputs)

    def freeze(self, filename, output_nodes_names=None):
        """Freezes the state of a session into a pruned computation graph.
        Creates a new computation graph where variable nodes are replaced by
        constants taking their current value in the session. The new graph will be
        pruned so subgraphs that are not necessary to compute the requested
        outputs are removed.
        @param session The TensorFlow session to be frozen.
        @param keep_var_names A list of variable names that should not be frozen,
                            or None to freeze all the variables in the graph.
        @param output_names Names of the relevant graph outputs.
        @param clear_devices
        @return The frozen graph definition.
        """
        assert filename[-3:] == ".pb", "You must provide a *.pb file"
        output_nodes_names = output_nodes_names or list(self.outputs.keys())
        # Remove the device directives from the graph for better portability
        graph_definition = self.graph.as_graph_def()
        for node in graph_definition.node: # pylint: disable=no-member
            node.device = ""
        frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(self.session, graph_definition, output_nodes_names)
        tf.train.write_graph(frozen_graph, os.path.dirname(filename), os.path.basename(filename), as_text=False)

        # compute the number of flops
        opt = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(self.graph, options=opt)
        total_flops = flops.total_float_ops
        print("frozen graph saved in '{}' with {} flops".format(filename, total_flops))
