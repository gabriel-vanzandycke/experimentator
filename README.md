# Experimentator

This framework is a ML experimentation library that offers a **modular implemation workflow** where a single python class holds the dataset, callacks and data processing graphs. By implementing a derived task-specific class, users can alter any attribute of the experiment instance.

## Installation and dependencies

This library is still in a beta phase and should be installed by cloning the repository:
```bash
git clone git@github.com:gabriel-vanzandycke/experimentator.git
cd experimentator and pip install -e .
```

It is deep-learning framework agnostic but, to this day, only the API for TensorFlow is implemented.

## Usage

Instantiation of an experiment requires a **configuration** that can be any dictionary, but it's recommended to work woth [`Pyconfyg`](https://github.com/gabriel-vanzandycke/pyconfyg) which handles python language configuration files. 

```python
from experimentator import BaseExperiment
class SpecificExperiment(BaseExperiment):
    pass
exp = SpecificExperiment({})
```

### Configuration file

The configuration dictionnary must define the following attributes:

**Generic attributes:**
- `experiment_type`: a list of classes that the experiment will instantiate. This enables decoupling features into independant class definitions like `AsyncExperiment` featuring asynchronous data loading or `CallbackedExperiment` featuring callbacks called before and after each batch, subset and epoch.
- `subsets`: a list of `experimentator.Subset`s (typically `training`, `validation` and `testing`), built from the dataset(s). The dataset(s) must inherit from [`mlworkflow.Dataset`](https://github.com/ispgroupucl/mlworkflow). Dataset items must be dictionnaries of named tensors, including model's input, targets and possibly any additional tensor required for training or evaluation.
- `batch_size`: an integer defining the batch size.
- `chunk_processors`: a list of operations applied on batches dataset items. Those items, called `chunk`, are successively processed by each chunk processor. After executing all chunk processors, the chunk dictionnary should contain a `loss` attribute used for the gradient descent algorithm.

**Tensorflow specific attributes:**
- `optimizer`: a `tf.keras.optimizers.Optimizer`.

### Tensorflow specific requirements
When using the TensorFlow implementation, the specific experiment must define the following attributes:
- `batch_inputs_names`: The initial chunk attribute names extracted from the input batch of data and converted to `tf.keras.Input`.
- `batch_metrics_names`: The chunk attribute names used to build the evaluation graph.
- `batch_outputs_names`: The chunk attribute names used to build the inference graph.

## Example

An example can be found in the `examples` folder:
```python
exp = build_experiment("configs/mnist.py")
exp.train(10) # trains for 10 epochs
```


# Acknowledgements, motivations and recommendations
This library was developed in many iterations since 2017, before the modern deep learning libraries became standards. It was greatly inspired by what became [ModulOM](https://openreview.net/forum?id=264iXDLnD59) from [Maxime Istasse](https://github.com/mistasse).

Overall, it was developped to meet the following requirements:
- Handling multiple inputs models, relevant for siameese training.
- Handling multiple outputs models, typically for multi-task learning.
- Modular graph construction with easy block substitutions instead of having a fixed graph.
- Clean implementation structure where task specific objects (callback, chunk processors, specific experiment class) can be implemented in a single file.

Desipite the numberous hours spent in this library developement and maintenance, I wouldn't recommend using it, as modern library offer much better support with similar features.
