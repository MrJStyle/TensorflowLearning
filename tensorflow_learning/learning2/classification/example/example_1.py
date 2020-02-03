import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.python.framework.ops import Tensor


class Mnist(object):


    def gen_data(self):
        (x, y), (x_val, y_val) = datasets.mnist.load_data()

        x: Tensor = 2 * tf.convert_to_tensor(x, dtype=tf.float32) / 255. - 1
        y: Tensor = tf.convert_to_tensor(y, dtype=tf.int32)

        print(f"x_shape: {x.shape} y_shape: {y.shape}")

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        train_dataset = train_dataset.batch(512)

