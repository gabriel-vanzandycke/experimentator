 

# Limitations (tested with TensorFlow v2.5)

## Impossible to create a random tensor of unknown shape
When saving a model with unknown shape, creation of a random tensor of the same dimension is not supported. 
For instance, a layer that adds a constant noise to a batch of images:
```
class AvoidLocalEqualities(tf.keras.layers.Layer):
    def build(self, input_shape):
        self.random_tensor = tf.expand_dims(tf.random.normal(input_shape[1:], mean=0, stddev=0.001), 0)
    def call(self, input_tensor):
        return self.random_tensor+input_tensor
```

This layer requires knowing the input tensor shape. However, it may be useful to **support unknown shapes** that results from building the (fully convolutional) network for arbitrary input shapes, in order to save the model with the `tf.keras.Model.save()` API.

> [...] defer weight creation to the first `__call__()` [...] wrapped in a `tf.init_scope`.

suggested in the [documentation on custom layers](https://www.tensorflow.org/guide/keras/custom_layers_and_models) doesn't help because dimensions are required to build the layer.

I tried using a `tf.Variable` with `validate_shape=False` but any random initializer seem to require a known shape, and a constant initializer defeats the purpose of the layer.

Any attempts resulted to one of the following errors:
```
ValueError: Cannot convert a partially known TensorShape to a Tensor
```
```
ValueError: None values not supported.
```

This was also [asked on StackOverflow](https://stackoverflow.com/questions/70202763/initialize-tensorflow-custom-layer-attribute-at-runtime), but without success.

## Item assignment not supported
```python
>>> x = tf.constant([0, 0, 0])
>>> x[0] = 4
TypeError: 'tensorflow.python.framework.ops.EagerTensor' object does not support item assignment
```

## Overcomplicated print of tensors
```python
x = tf.keras.Input(shape=(None, None, 3))
y = tf.sigmoid(x)
def f(a): tf.print(a); return a  # Impossible to use - builtin `print`
y = tf.keras.layers.Lambda(f)(y) #                   - neither `tf.print`
model = tf.keras.Model(x, y)
result = model(np.random.random((10, 20, 3))*20-10)
```

## Impossible to release memory
It is currently impossible to release memory in the current thread. Workarounds are:

- Launch jobs in different threads with multiprocessing.
- Kill manually unix processes using the GPU.
- Release memory directly with CUDA (which can leave the GPU in an unstable state)

```python
import multiprocessing
p = multiprocessing.Process(target=train_function, args=(a,), kwargs={})
p.start()
p.join()
```
See https://github.com/tensorflow/tensorflow/issues/36465

## Complicated limitations of tf.function
See



# Additional notes

## Arguments of @tf.function functions should not be python native types

Graphs created from `tf.function`-decorated functions are built using the input parameters **value**. When using Python native types `(1.0 == 1)` a **new graph** is created for every different input parameter value with a **huge drop in performance**. We should always feed those functions with `tf.Tensors`.

https://pgaleone.eu/tensorflow/tf.function/2019/04/03/dissecting-tf-function-part-2/

## Remove first pooling layer from TensorFlow pre-trained ResNet50

This modification is inspired by PifPaf ResNet backbone: the first 3x3 pooling layer is removed from the pre-trained fully convolutional ResNet50. Thanks for Piotr Graszka for the help.
```python
base_model = tf.keras.applications.ResNet50(include_top=False, input_shape=(None, None, 3))
head_model = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[4].output)
tail_input = tf.keras.Input(tensor=base_model.layers[7].input)
tail_model = tf.keras.Model(inputs=tail_input, outputs=base_model.output)
model = tf.keras.Sequential([head_model, tail_model])
```

## Print intermediate tensor from graph with TensorFlow 2.x
When creating a TensorFlow 2 model, the `tf.print` function cannot be used on the graph operations and would raises the following error:
```python
TypeError: Cannot convert a symbolic Keras input/output to a numpy array. This error
may indicate that you're trying to pass a symbolic value to a NumPy call, which is
not supported. Or, you may be trying to pass Keras symbolic inputs/outputs to a TF
API that does not register dispatching, preventing Keras from automatically
converting the API call to a lambda layer in the Functional Model.
```

To circumvent this limitation, the `tf.print` call can be wrapped within a `tf.keras.layers.Lambda` layer. The tensor to be printed needs to be returned by the function to integrate the `tf.print` call into the graph definition.

```python
import tensorflow as tf

def print_tensor(x):
    tf.print(x)
    return x

inputs = tf.keras.Input(dtype=tf.uint8, shape=(10,20,3))
x = tf.keras.layers.Conv2D(10, 3)(inputs)
x = tf.keras.layers.Lambda(print_tensor)(x)  # allows printing tensor at runtime
outputs = tf.keras.layers.MaxPool2D(2, 2)(x)
model = tf.keras.Model(inputs, outputs)

result = model(np.ones((8,10,20,3)))         # intermediate values is printed
```

## Save input from previous session execution with TensorFlow 1.14

In the scenario where we want to provide two consecutive images to the neural network, we wan’t to avoid sending twice each image (once as first-image and the second time as second-image).

Here is a minimal code-block that answer this question with static graph on TensorFlow 1.14

### Graph definition and initialisation
```python
import tensorflow as tf
tf.reset_default_graph()

input2 = tf.Variable(0, dtype=tf.uint8)
input1 = tf.placeholder_with_default(input2, shape=(), name="input_value")

# Main computation done on input_image and previous_image
result = input1 - input2

# Operations required to save input1 for later be input2
tmp_value = tf.Variable(0, dtype=tf.uint8)
assign_input1_to_tmp = tf.assign(tmp_value, input1)
assign_tmp_to_input2 = tf.assign(input2, tmp_value)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
```

### Session Execution
```python
# First execution, to retrieve the result
input_value = 42
fetches = {
    "result": result, # Main computation
    "assign_input1_to_tmp": assign_input1_to_tmp # Save input1 in a temporary variable
}
print(dict(zip(fetches.keys(), sess.run(feed_dict={input1: input_value}, fetches=list(fetches.values())))))

# Second execution, to move the temporary variable value into input2 variable
fetches = {
    "assign_tmp_to_input2": assign_tmp_to_input2
}
print(dict(zip(fetches.keys(), sess.run(feed_dict={}, fetches=list(fetches.values())))))
```
**Note:** I couldn’t succeed to make it work with one single session execution and avoid the `tmp_value` variable.

## Device control for TensorFlow 1.14 operations
TensorFlow 1.14 can put operations on the CPU without notice! It means that some computation is done on the CPU instead of the GPU, but also that data is transferred between GPU memory and CPU memory, which has a huge cost.

To force the use of the GPU, you can use the following code:
```python
with tf.device("/gpu:0"):
  # tensorflow gpu operators
```
It happens that some operators don’t have a GPU implementation ! It can be because the data type is not implemented (eg: `tf.layers.max_pool2D` is not implemented for type `uint8` on the GPU.

If there is a XLA_GPU implementation, you can force it with the following code:
```python
with tf.device("/gpu:0"):
  # tensorflow GPU operators
  with tf.device("/device:XLA_GPU:0"):
    # tensorflow XLA_GPU operators
  # tensorflow GPU operators
```
Else, you should probably use another function.

I check that no operator is executed on the CPU with the following code:
```python
graph = tf.Graph()
sess = tf.Session(tf.ConfigProto(log_device_placement=True), graph=graph)
graph_nodes_on_cpu = [n for n in graph.as_graph_def().node if "CPU" in n.device]
assert not graph_nodes_on_cpu, "One on more node(s) are defined on the CPU {}".format(graph_nodes_on_cpu)
```

## Profiling TensorFlow 1.14 static session execution
Good readings https://aistein.github.io/dlprof/

## Spacial Pyramid Pooling with unknown shape
Usually, (Spacial Pyramid Pooling)[https://arxiv.org/abs/1406.4729] is implemented using a conventional pooling layer with the kernel size and stride took from the input tensor shape, and a given scale `s`:
```python
def avg_spp_layer(inputs, s):
  h, w = inputs.get_shape().as_list()[1:3]
  size = (h, w)/s
  pooled = tf.layers.average_pooling2d(inputs, pool_size=size, strides=size))  # spac. pyr. pool [B, s, s, C]
  return tf.images.resize_bilinear(pooled, (h, w))                             # original size   [B, H, W, C]
```

However, this implementation doesn’t work when the input tensor has unknown shape (eg. when it depends on another input). Indeed, the pooling layer needs to know in advance the pool_size and the strides that must be provided as integers, and not as `tf.Tensor`.

To address this issue, I came up with a different implementation that computes the pyramid pooling per region by splitting the input with a binary mask, allowing to address each pooling region separately.

The actual mask is obtained by rescaling a simple s×s mask having s*s channels (illustrated below for scale=2) using nearest neighbor.

![image](https://user-images.githubusercontent.com/18050620/205335123-92f6bcdb-4aa4-4d13-b34b-5f40b4aa9796.png)

This implementation requires to use an additional dimension for the mask. The tensor shapes are shown in comment to help understand the algorithm. The max spacial pyramid pooling is achieved with a very similar implementation.

```python
def avg_spp_layer(self, input, size, name, padding=DEFAULT_PADDING):   #                    tensor dim
  eye = tf.eye(size*size, batch_shape=(tf.shape(input)[0],))           # identity matrix   [B, s*s, s*s]
  mask = tf.reshape(eye, (-1, size, size, size*size))                  # simple mask       [B, s, s, s*s]
  mask = tf.image.resize_nearest_neighbor(mask, tf.shape(input)[1:3])  # full mask         [B, H, W, s*s]
  spp = tf.multiply(tf.expand_dims(input, 4), tf.expand_dims(mask, 3)) # splitted image    [B, H, W, C, s*s]
  spp = tf.reduce_mean(spp, axis=[1,2])*size*size                      # average           [B, 1, 1, C, s*s]
  spp = tf.reshape(spp, (-1, tf.shape(input)[3], size, size))          # reshaping         [B, C, s, s]
  spp = tf.transpose(spp, [0,2,3,1], name=name)                        # transposing       [B, s, s, C]
  return tf.images.resize_bilinear(pooled, (h, w))                     # original size     [B, H, W, C]
```

 
 
