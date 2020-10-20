import os
import random
import logging
import numpy as np
from .base_experiment import BaseExperiment, lazyproperty
from .utils import OutputInhibitor
if False: # pylint: disable=using-constant-test
    lazyproperty = property

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
with OutputInhibitor("Tensorflow lib"):
    import tensorflow.python.autograph.utils.ag_logging # pylint: disable=no-name-in-module
    import tensorflow as tf
    tf.get_logger().setLevel(logging.ERROR)

class TensorflowExperiment(BaseExperiment):
    run_options = None  # overwritten by callbacks
    run_metadata = None # overwritten by callbacks

    @lazyproperty
    def graph(self):
        with OutputInhibitor():
            graph = tf.Graph()
        return graph

    def init_model(self, weights=None):
        # pylint: disable=pointless-statement
        self.process # as a lazyproperty (no need to use the parenthisis)
        with self.graph.as_default():  # pylint: disable=not-context-manager
            self.optimization_step
            self.metrics
            random_seed = int(self["manager_id"].replace("_","")[4:])
            tf.set_random_seed(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
            with tf.name_scope("initialization"):
                self.session.run(tf.local_variables_initializer())
                self.session.run(tf.global_variables_initializer())
            graph_nodes_on_cpu = [n for n in self.graph.as_graph_def().node if "CPU" in n.device] # pylint: disable=no-member
            assert not graph_nodes_on_cpu, "One on more node(s) are defined on the CPU {}".format(graph_nodes_on_cpu)
        if weights is None and "weights" in self.logger:
            try:
                weights = self.logger["weights"]
            except FileNotFoundError:
                print("Last weights have been deleted. Using random weights.")
        if weights is not None:
            self.set_weights(weights)

    @lazyproperty
    def session(self):
        # TODO: check for "device_filters", "gpu_options", "graph_options", ... in config
        log_device_placement = self.get("log_device_placement", False)
        allow_soft_placement = self.get("allow_soft_placement", False)
        config = tf.ConfigProto(log_device_placement=log_device_placement, allow_soft_placement=allow_soft_placement)
        config.gpu_options.visible_device_list = self["GPU"]
        config.gpu_options.allow_growth = True # pylint: disable=no-member
        with OutputInhibitor():
            sess = tf.Session(config=config, graph=self.graph)
        return sess

    @lazyproperty
    def loss(self):
        return self.chunk["loss"]

    @lazyproperty
    def optimizer(self):
        with self.graph.as_default():  # pylint: disable=not-context-manager
            with self.graph.device(self.get("device", "/gpu:0")): # pylint: disable=not-context-manager
                with tf.name_scope("optimizer"):
                    optimizer = self["optimizer"]()
        return optimizer

    @lazyproperty
    def optimization_step(self):
        try:
            return self.optimizer.minimize(self.loss, var_list=None) # None = all variables
        except KeyError:
            return None

    # set weights from variable
    def set_weights(self, weights):
        print("Loading weights...", end="")
        with self.graph.as_default():  # pylint: disable=not-context-manager
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        self.session.run([k.assign(v) for k, v in zip(var_list, weights)])
        print(" Done.")

    # get weights from variable
    def get_weights(self):
        with self.graph.as_default():  # pylint: disable=not-context-manager
            var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        return self.session.run(var_list)

    @property
    def learning_rate(self):
        # pylint: disable=protected-access
        # TODO: try 'self.graph.get_operation_by_name("Adam/learning_rate")'
        return 42#self.session.run(self.optimizer._lr_t) # probably only works with Adam # doesn't seem to work

    @property
    def weights(self):
        return self.get_weights()

    @lazyproperty # done only once in tensorflow: to create the graph
    def process(self):
        with self.graph.as_default():  # pylint: disable=not-context-manager
            with self.graph.device(self.get("device", "/gpu:0")):  # pylint: disable=not-context-manager
                for ChunkProcessor in self["chunk_processors"]:
                    if ChunkProcessor is None:
                        continue
                    chunk_processor = ChunkProcessor()
                    chunk_processor.graph = self.graph
                    with tf.name_scope(chunk_processor.__class__.__name__):
                        chunk_processor(self.chunk)

    @lazyproperty
    def chunk(self):
        return self.placeholders.copy()

    def feed_dict(self, data):
        # this codes creates the error
        # int() argument must be a string, a bytes-like object or a number, not 'dict'
        # def preprocess(value, placeholder):
        #     if isinstance(placeholder, tf.SparseTensor):
        #         indices = np.where(value)
        #         value[indices]
        #         return indices, value[indices], value.shape
        #     else:
        #         return data
        # return {self.placeholders[name]: preprocess(value, self.placeholders[name]) for name, value in data.items() if name in self.placeholders}
        return {self.placeholders[name]: value for name, value in data.items() if name in self.placeholders}


    def batch_train(self, data):
        fetches = {
            **self.metrics,
            "loss": self.loss,
            "optimization_step": self.optimization_step
        }
        feed_dict = self.feed_dict({"is_training": True, **data})

        results = self.session.run(list(fetches.values()), feed_dict=feed_dict, options=self.run_options, run_metadata=self.run_metadata)

        results = dict(zip(fetches.keys(), results))
        results.pop("optimization_step")
        return results

    def batch_eval(self, data, **fetches):
        fetches = {
            **self.metrics,
            #"loss": self.loss,
            # this is necessary when I want both the metrics and the result
            **self.outputs,
            **fetches
        }
        # TODO: use sess.make_callable instead of sess.run ?
        feed_dict = self.feed_dict({"is_training": False, **data})
        results = self.session.run(list(fetches.values()), feed_dict=feed_dict, options=self.run_options, run_metadata=self.run_metadata)
        results = dict(zip(fetches.keys(), results))
        return results

    def batch_infer(self, data):
        fetches = {
            **self.outputs
        }
        feed_dict = self.feed_dict({"is_training": False, **data})
        results = self.session.run(list(fetches.values()), feed_dict=feed_dict)
        results = dict(zip(fetches.keys(), results))

        return results

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
        output_nodes_names = output_nodes_names or list(self.outputs.keys())
        with self.graph.as_default(): # pylint: disable=not-context-manager
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
