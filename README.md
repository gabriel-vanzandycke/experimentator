# Experimentator

This framework is a ML experimentation library that offers a **modular implemation** workflow where a single python class holds the dataset, callacks and data processing graphs. By implementing a derived task-specific class, users can alter any attribute of the experiment instance.

## Installation and dependencies

This library is still in a beta phase and should be installed by cloning this repository:
```bash
git clone git@github.com:gabriel-vanzandycke/experimentator.git
cd experimentator and pip install -e .
```

The library is deep-learning framework agnostic but, to this day, only the API for TensorFlow is implemented.

## Usage

Instantiation of an experiment requires a **configuration** that can be any dictionary, but it's recommended to work woth [`Pyconfyg`](https://github.com/gabriel-vanzandycke/pyconfyg) which handles python language configuration files. It must define the following attributes:

### Generic attributes:
- `experiment_type`: a list of classes that the experiment will instantiate. This enables decoupling features into independant class definitions like `AsyncExperiment` that features asynchronous data loading or `CallbackedExperiment` that features callbacks before and after each batch, subset and epoch.
- `subsets`: a list of `experimentator.Subset`s (typically `training`, `validation` and `testing`), built from the dataset(s). The dataset(s) must inherit from [`mlworkflow.Dataset`](https://github.com/ispgroupucl/mlworkflow).
- `batch_size`: an integer defining the batch size.
- `chunk_processors`: a list of operations applied on batches of data provided as dictionnaries of named tensors. Those dictionnaries, called `chunk`, are successively processed by each chunk processor. After this operation the chunk should contain a `loss` attribute on which the gradient descent will be performed.

### Tensorflow specific attributes:
- `optimizer`: a `tf.keras.optimizers.Optimizer`.



** Configuration file**
```python
data = tf.keras.datasets.mnist.load_data()
```


```python
from functools import cached_property
from experimentator import BaseExperiment

class SpecificExperiment(BaseExperiment):
    @cached_property
    def subsets(self):
        subsets = super().subsets
        # <custom operations over subsets>
        return subsets
    def 
exp = type("Exp", (SpecificExperiment, BaseExperiment), config)
```
goals:
- allow multi input, multi output model arguments
- plug and play GPU operators instead of a fixed flow to create model (@maxime)
- modular implementation (callbacks, GPU operators, dataset transformations) such that common objects/functions can be shared and specific objects/functions are grouped per task (@maxime)
- single python object through which all parameters/configuration can be read or modified (@maxime)
- 
