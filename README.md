# Experimentator - a ML experimentation library

This framework offers a **modular implemation** workflow where a single python class holds the dataset, callacks and data processing graphs. By implementing a derived task-specific class, users can alter any attribute of the experiment instance.

## Installation and dependencies

This library is still in a beta phase and should be installed by cloning this repository:
```bash
git clone git@github.com:gabriel-vanzandycke/experimentator.git
cd experimentator and pip install -e .
```

The library is deep-learning framework agnostic but, to this day, only the API for TensorFlow support was implemented.

## Usage

The **configuration** require to instantiate an experiment class can be any dictionary, but it's recommended to work woth [`Pyconfyg`](https://github.com/gabriel-vanzandycke/pyconfyg) which handles python language configuration files.

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
