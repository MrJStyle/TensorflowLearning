import numpy as np
import tensorflow as tf

from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.variables import Variable


class Graph(object):
    a = tf.constant([[1.0, 6.0], [3.0, 5.0]], name="a")
    b = tf.constant([[3.0, 4.0], [5.0, 7.0]], name="b")

    def add(self) -> Tensor:
        """
        tf中张量加法的使用
        Returns: Tensor

        """
        with tf.Session() as sess:
            result: Tensor = self.a + self.b
            return sess.run(result)

    def mul(self) -> np.array:
        """
        tf中张量乘法的使用
        Returns: Tensor

        """
        with tf.Session() as sess:
            result: Tensor = tf.matmul(self.a, self.b)
            return sess.run(result)

    @staticmethod
    def init_var() -> np.array:
        """
        tf.Variable变量使用前需要initializer
        Returns: np.array

        """
        w1: Variable = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
        with tf.Session() as sess:
            sess.run(w1.initializer)
            return w1.eval()

    @staticmethod
    def init_all_var():
        """
        全局初始化tf.Variable
        Returns:

        """
        w1: Variable = tf.Variable(tf.random.normal([2, 3], stddev=1, seed=1))
        w2: Variable = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
        with tf.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            print(f"[init_all_var]: w1={w1.eval()} w2={w2.eval()}")

    def clip_by_value(self):
        """
        tf.clip_by_value可将一个张量中的数值限制在一个范围之内
        Returns:

        """
        y = tf.clip_by_value(self.a, 3, 5)
        with tf.Session() as sess:
            print(sess.run(y))

    def log(self):
        """
        求对数
        Returns:

        """
        y = tf.math.log(self.a)
        with tf.Session() as sess:
            print(sess.run(y))

    def select_and_greater(self):
        sess = tf.compat.v1.InteractiveSession()
        print(tf.greater(self.a, self.b).eval())

        print(tf.where(tf.greater(self.a, self.b), self.a, self.b).eval())
        sess.close()

    @staticmethod
    def regularization():
        weights = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
        with tf.Session() as sess:
            print(sess.run(tf.keras.regularizers.l1(0.5)(weights)))

            print(sess.run(tf.keras.regularizers.l2(0.5)(weights)))


if __name__ == "__main__":
    g = Graph()
    print(g.add)
    x = g.mul()
    print(type(x))
    print(g.init_var())
    g.init_all_var()
    g.clip_by_value()
    g.log()
    g.select_and_greater()
    g.regularization()