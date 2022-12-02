 

# Limitations (tested with TensorFlow v2.5)

## Impossible to create a random tensor of unknown shape
When saving a model with unknown shape, creation of a random tensor of the same dimension is not supported. See https://stackoverflow.com/questions/70202763/initialize-tensorflow-custom-layer-attribute-at-runtime

## Item assignment not supported
>>> x = tf.constant([0, 0, 0])
>>> x[0] = 4
TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment

## Overcomplicated print of tensors
x = tf.keras.Input(shape=(None, None, 3))
y = tf.sigmoid(x)
def f(a): tf.print(a); return a  # Impossible to use - builtin `print`
y = tf.keras.layers.Lambda(f)(y) #                   - neither `tf.print`
model = tf.keras.Model(x, y)
result = model(np.random.random((10, 20, 3))*20-10)

## Impossible to release memory
It is currently impossible to release memory in the current thread. Workarounds are:

Launch jobs in different threads with multiprocessing.

Kill manually unix processes using the GPU.

Release memory directly with CUDA (which can leave the GPU in an unstable state)

import multiprocessing
p = multiprocessing.Process(target=train_function, args=(a,), kwargs={})
p.start()
p.join()
See https://github.com/tensorflow/tensorflow/issues/36465

## Complicated limitations of tf.function
See



 
