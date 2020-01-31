import numpy as np
import tensorflow as tf

from typing import List
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.variables import Variable
from numpy.random.mtrand import RandomState


class Example(object):
    batch_size: int = 8

    def main(self):
        # 两个节点输入
        x: Tensor = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
        # 回归问题一般只有一个输出节点
        y_: Tensor = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

        # 定义一个单层的神经网络前向传播过程，这里仅是简单的加和
        w1: Variable = tf.Variable(tf.random.normal([2, 1], stddev=1, seed=1))
        y: Tensor = tf.matmul(x, w1)

        loss_less: int = 10
        loss_more: int = 1
        loss = tf.reduce_sum(
            tf.where(
                tf.greater(y, y_),
                (y - y_) * loss_more,
                (y_ - y) * loss_less
            )
        )

        train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        rdm: RandomState = RandomState(1)

        data_set_size = 128

        X: np.array = rdm.rand(data_set_size, 2)

        Y: List = [[x1 + x2 + rdm.rand() / 10.0 - 0.05] for x1, x2 in X]

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            steps = 5000

            for i in range(steps):
                start: int = (i * self.batch_size) % data_set_size
                end: int = min(start + self.batch_size, data_set_size)

                sess.run(
                    train_step,
                    feed_dict={x: X[start: end], y_: Y[start: end]}
                )

                print(sess.run(w1))


if __name__ == "__main__":
    e = Example()
    e.main()





