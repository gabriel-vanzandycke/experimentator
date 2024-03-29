import torch
from experimentator import ChunkProcessor

class DoNothing(ChunkProcessor):
    def __call__(self, chunk):
        pass

class CastFloat(ChunkProcessor):
    def __init__(self, tensor_names):
        self.tensor_names = [tensor_names] if isinstance(tensor_names, str) else tensor_names
    def __call__(self, chunk):
        for tensor_name in self.tensor_names:
            if tensor_name in chunk:
                chunk[tensor_name] = chunk[tensor_name].float()

class RenameTensor(ChunkProcessor):
    def __init__(self, **mapping):
        self.mapping = mapping
    def __call__(self, chunk):
        for name in self.mapping:
            if name in chunk:
                chunk[self.mapping[name]] = chunk[name]

class Normalize(ChunkProcessor):
    def __init__(self, tensor_names):
        self.tensor_names = [tensor_names] if isinstance(tensor_names, str) else tensor_names
    def __call__(self, chunk):
        for tensor_name in self.tensor_names:
            if tensor_name in chunk:
                assert isinstance(chunk[tensor_name].dtype, (torch.FloatTensor, torch.DoubleTensor))
                chunk[tensor_name] = chunk[tensor_name]/255

# class BatchStandardize(ChunkProcessor):
#     def __init__(self, tensor_names):
#         self.tensor_names = [tensor_names] if isinstance(tensor_names, str) else tensor_names
#     def __call__(self, chunk):
#         for tensor_name in self.tensor_names:
#             if tensor_name in chunk:
#                 chunk[tensor_name] = tf.image.per_image_standardization(chunk[tensor_name])

# class DatasetStandardize(ChunkProcessor):
#     def __init__(self, tensor_names, mean, std):
#         self.tensor_names = [tensor_names] if isinstance(tensor_names, str) else tensor_names
#         self.mean = mean
#         self.std = std
#     def __call__(self, chunk):
#         for tensor_name in self.tensor_names:
#             if tensor_name in chunk:
#                 chunk[tensor_name] = (chunk[tensor_name]-self.mean)/self.std

# class Sigmoid(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["batch_sigmoid"] = tf.nn.sigmoid(chunk["batch_logits"])

# class SigmoidCrossEntropyLoss(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["loss"] = tf.reduce_mean(tf.keras.losses.binary_crossentropy(chunk["batch_target"][...,tf.newaxis], chunk["batch_logits"], True), axis=[0,1,2])

# class SigmoidCrossEntropyLossMap(ChunkProcessor):
#     def __call__(self, chunk):
#         batch_target = chunk["batch_target"] if len(chunk["batch_target"].shape) == 4 else tf.expand_dims(chunk["batch_target"], -1)
#         chunk["loss_map"] = tf.keras.losses.binary_crossentropy(batch_target, chunk["batch_logits"], from_logits=True)

# class GlobalMaxPoolingLogits(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["batch_logits"] = tf.reduce_max(chunk["batch_logits"], axis=[1,2])

# class GlobalAvgPoolingLogits(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["batch_logits"] = tf.reduce_mean(chunk["batch_logits"], axis=[1,2])


# class OneHot(ChunkProcessor):
#     def __init__(self, tensor_name, num_classes):
#         self.num_classes = num_classes
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         assert len(chunk[self.tensor_name].shape) == 3, "Wrong shape for 'batch_target'. Expected (B,H,W). Received {}".format(chunk[self.tensor_name].shape)
#         chunk[self.tensor_name] = tf.one_hot(chunk[self.tensor_name], self.num_classes)

# class Argmax(ChunkProcessor):
#     def __init__(self, tensor_name):
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         chunk[self.tensor_name] = tf.argmax(chunk[self.tensor_name], axis=-1)


# # from tf1, untested

# class UnSparsify(ChunkProcessor):
#     def __call__(self, chunk):
#         for name in chunk:
#             if isinstance(chunk[name], tf.SparseTensor):
#                 chunk[name] = tf.sparse.to_dense(chunk[name])


# class DeNormalize(ChunkProcessor):
#     def __init__(self, tensor_name):
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         chunk[self.tensor_name] = tf.saturate_cast(chunk[self.tensor_name]*255, tf.uint8, name=self.tensor_name)

# class Classify(ChunkProcessor):
#     def __init__(self, tensor_name="batch_softmax"):
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         chunk["batch_output"] = tf.cast(tf.argmax(chunk[self.tensor_name], axis=-1, output_type=tf.int32), tf.uint8)

# class Output(ChunkProcessor):
#     def __init__(self, tensor_name):
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         chunk["batch_output"] = tf.identity(chunk[self.tensor_name], name="batch_output")

# class Heatmap(ChunkProcessor):
#     def __init__(self, tensor_name, class_index):
#         self.class_index = class_index
#         self.tensor_name = tensor_name
#     def __call__(self, chunk):
#         chunk["batch_heatmap"] = chunk[self.tensor_name][:,:,:,self.class_index]

# class SoftmaxCrossEntropyLoss(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["loss"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(chunk["batch_target"], chunk["batch_logits"]))

# class Softmax(ChunkProcessor):
#     def __call__(self, chunk):
#         assert chunk["batch_logits"].shape[-1] > 1, "It doesn't mean anything to take the softmax of a single output feature map"
#         chunk["batch_softmax"] = tf.nn.softmax(chunk["batch_logits"])

# class OneHotTarget(OneHot):
#     def __init__(self, *args, **kwargs):
#         super().__init__(tensor_name="batch_target", *args, **kwargs)

# class StopGradients(ChunkProcessor):
#     def __init__(self, exceptions=None):
#         self.exceptions = exceptions or []
#     def __call__(self, chunk):
#         for name in chunk:
#             if name not in self.exceptions:
#                 chunk[name] = tf.stop_gradient(chunk[name])

# class SoftmaxCrossEntropyLossMap(ChunkProcessor):
#     def __init__(self, label_smoothing=0):
#         self.label_smoothing = label_smoothing
#     def __call__(self, chunk):
#         if "batch_softmax" in chunk:
#             chunk["loss_map"] = tf.losses.log_loss(labels=chunk["batch_target"], predictions=chunk["batch_softmax"], reduction=tf.losses.Reduction.NONE)
#         else:
#             chunk["loss_map"] = tf.losses.softmax_cross_entropy(onehot_labels=chunk["one_hot"], logits=chunk["batch_logits"], reduction=tf.losses.Reduction.NONE, label_smoothing=self.label_smoothing)

# class MeanSquaredErrorLossMap(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["loss_map"] = tf.losses.mean_squared_error(chunk["batch_target"], chunk["batch_logits"], reduction=tf.losses.Reduction.NONE)

# class ReduceLossMap(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["loss"] = tf.reduce_mean(chunk["loss_map"])

# class KerasHack(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["batch_logits"] = tf.subtract(chunk["batch_logits"], tf.expand_dims(tf.reduce_max(chunk["batch_logits"], axis=-1), -1))

# class MeanSquareErrorLoss(ChunkProcessor):
#     def __call__(self, chunk):
#         chunk["loss"] = tf.losses.mean_squared_error(chunk["batch_target"], chunk["batch_logits"])
