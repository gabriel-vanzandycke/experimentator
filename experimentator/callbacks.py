import time
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline # pylint: disable=no-name-in-module


class Callback():
    precedence = 10
    def __init__(self, *args, **kwargs):
        pass
    def fire(self, event, state):
        cb = getattr(self, "on_{}".format(event), None)
        if cb:
            cb(**state, state=state) # pylint: disable=not-callable

class InitState(Callback):
    def on_epoch_begin(self, state, **_):
        for key in [k for k in state if k!= "epoch"]:
            state[key] = np.nan

class LogState(Callback):
    precedence = 100 # very last
    subset_names = set()
    def __init__(self, exp):
        self.logger = exp.logger
    def on_cycle_end(self, subset_name, state, **_):
        self.subset_names.add(subset_name)
        subset_metrics = dict(state) # dict() makes a copy of "state"
        self.logger["{}_metrics".format(subset_name)] = subset_metrics
    def on_epoch_begin(self, state, **_):
        self.logger["metrics"] = list(state.keys())
        # clear metrics
        for subset_name in self.subset_names:
            subset_metrics = dict(state)
            self.logger["{}_metrics".format(subset_name)] = subset_metrics

class AverageLoss(Callback):
    def on_cycle_begin(self, **_):
        self.loss = []
    def on_batch_end(self, loss, **_):
        self.loss.append(loss)
    def on_cycle_end(self, state, **_):
        state["loss"] = np.mean(self.loss)

class MeasureTime(Callback):
    def on_epoch_begin(self, **_):
        self.tic_epoch = time.time()
    def on_cycle_begin(self, **_):
        self.history = []
        self.tic_cycle = time.time()
    def on_batch_begin(self, **_):
        self.tic_batch = time.time()
    def on_batch_end(self, **_):
        toc_batch = time.time()
        self.history.append(toc_batch - self.tic_batch)
    def on_cycle_end(self, state, **_):
        toc_cycle = time.time()
        state["batch_time"] = np.mean(self.history)
        state["cycle_time"] = toc_cycle - self.tic_cycle
    def on_epoch_end(self, state, **_):
        toc_epoch = time.time()
        state["epoch_time"] = toc_epoch - self.tic_epoch


class ProfileBatch(Callback):
    precedence = 95 # almost last
    def __init__(self, exp):
        self.loggdir = exp.get("loggdir", "profiler_logs")
        self.profiler = tf.profiler.Profiler(graph=exp.graph)
        self.exp = exp
        self.summary = tf.summary.FileWriter(self.loggdir + '/train', graph=exp.graph, session=exp.session)
        self.target_batch = 10

    def on_batch_begin(self, batch_id, **_):
        if batch_id != self.target_batch:
            return
        #tf.summary.trace_on(graph=True, profiler=True)
        self.exp.run_metadata = tf.RunMetadata()
        self.exp.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # pylint: disable=no-member
        # tf.profiler.experimental.start(
        #     self.loggdir, tf.profiler.ProfilerOptions(host_tracer_level=2)
        # )
    def on_batch_end(self, batch_id, subset_name, state, **_):
        key = "{}_{}_{}".format(self.exp["experiment_id"], subset_name, batch_id)
        if batch_id != self.target_batch:
            return

        if False:
            # with self.summary.as_default():
            #     for metric in self.exp.metrics.keys():
            #         tf.summary.scalar(metric, state[metric], step=batch_id)
            #     tf.summary.trace_export(
            #         name="func_trace",
            #         step=batch_id,
            #         profiler_outdir=self.loggdir
            #     )
            self.summary.add_run_metadata(self.exp.run_metadata, "run_metadata", batch_id)
            # for metric_name in self.exp.metrics.keys():
            #     tf.summary.scalar(metric_name, state[metric_name])
            #     self.summary.add_summary()#, step=batch_id))

            self.profiler.add_step(batch_id, self.exp.run_metadata)


            #self.profiler.profile_operations(opts)

            # opts = (tf.profiler.ProfileOptionBuilder()
            #     .with_max_depth(10)
            #     .with_min_micros(1000)
            #     .select(['accelerator_micros'])
            #     .with_stdout_output()
            #     .build())
            #     #.time_and_memory()
            #     #.with_stdout_output()
            #     #.build())

            # Or you can generate a timeline:
        
        opts = (tf.profiler.ProfileOptionBuilder(
                tf.profiler.ProfileOptionBuilder.time_and_memory())
                .with_step(batch_id)
                .with_timeline_output("{}_timeline.ctf".format(key)).build())
        self.profiler.profile_graph(options=opts)
        
        if True:
            opts = (tf.profiler.ProfileOptionBuilder(
                    tf.profiler.ProfileOptionBuilder.time_and_memory())
                    .with_step(batch_id)
                    .with_timeline_output("{}_timeline_python.ctf".format(key)).build())
            self.profiler.profile_python(options=opts)
            #tf.profiler.experimental.stop()
        #
        #
        if True:
            tl = timeline.Timeline(self.exp.run_metadata.step_stats) # pylint: disable=no-member
            with open("tf_timeline_{}".format(key), 'w') as f:
                f.write(tl.generate_chrome_trace_format())
        
        self.exp.run_metadata = None
        self.exp.run_options = None


class ProfileCycle_DOESNTWORK(Callback):
    precedence = 95 # almost last
    def __init__(self, exp, subset_name=None): # pylint: disable=super-init-not-called
        self.exp = exp
        self.subset_name = subset_name
        self.profiler = tf.profiler.Profiler(graph=exp.graph)

    def on_cycle_begin(self, subset_name, **_):
        if self.subset_name != subset_name:
            return

        self.exp.run_metadata = tf.RunMetadata()
        self.exp.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) # pylint: disable=no-member

    def on_cycle_end(self, subset_name, epoch, **_):
        if self.subset_name != subset_name:
            return
        opts = (tf.profiler.ProfileOptionBuilder(
                tf.profiler.ProfileOptionBuilder.time_and_memory())
                .with_step(0)
                .with_timeline_output("timeline_python_{}_cycle_{}.ctf".format(subset_name, 0)).build())
        self.profiler.profile_python(options=opts)

        self.exp.run_metadata = None
        self.exp.run_options = None
