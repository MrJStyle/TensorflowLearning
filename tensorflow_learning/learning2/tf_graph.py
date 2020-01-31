import numpy as np
import tensorflow as tf

from tensorflow.python.framework.ops import Tensor


class Graph(object):
    a = tf.constant([[1, 2], [3, 5]], name="a")
    b = tf.constant([[3, 4], [5, 7]], name="b")

    def add(self):
        print(self.a + self.b)

    def mul(self) -> np.array:
        with tf.Session() as sess:
            result: Tensor = tf.matmul(self.a, self.b)
            return sess.run(result)


if __name__ == "__main__":
    g = Graph()
    x = g.mul()
    print(type(x))
