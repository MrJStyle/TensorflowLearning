import tensorflow as tf


class TensorflowSession(object):
    a = tf.constant([1, 2], name="a")
    b = tf.constant([3, 4], name="b")
    result = a + b

    def run(self):
        with tf.Session() as sess:
            sess.run(self.result)


if __name__ == "__main__":
    t = TensorflowSession()
    t.run()