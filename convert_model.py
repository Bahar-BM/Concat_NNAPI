#!/usr/bin/python3

import tensorflow as tf
from tensorflow import python as tf_python
import numpy as np


def representative_data_concat():
    for _ in range(100):
      data = np.random.rand(1, 128, 128, 3)
      data1 = np.random.rand(1, 128, 128, 6)
      yield [data.astype(np.float32), data1.astype(np.float32)]

def representative_data_dense():
    for _ in range(100):
      data = np.random.rand(1, 4096)
      yield [data.astype(np.float32)]

######## Conversion - INT8 #########
tf_model = tf.keras.models.load_model('quantize_concat.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_concat
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'int8_quantize_concat.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

#################
tf_model = tf.keras.models.load_model('large_Dense.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_dense
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]


tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'int8_large_Dense.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)


