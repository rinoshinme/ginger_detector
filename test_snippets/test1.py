import tensorflow as tf


def test():
    size = 4
    a = tf.range(size, dtype=tf.int32)
    x = tf.tile(a[tf.newaxis, :], [size, 1])
    with tf.Session() as sess:
        xval, aval = sess.run([x, a])
        print(xval)
        print(aval)


if __name__ == '__main__':
    test()
