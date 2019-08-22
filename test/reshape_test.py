import tensorflow as tf

v = tf.constant([[1,2,3],[4,5,6]])
u = tf.reshape(v, [3,2])
with tf.Session() as sess:
    sess.run(u)
    print(u)