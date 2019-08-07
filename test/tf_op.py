
import tensorflow as tf
import numpy as np

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
#
# t3 = tf.concat([t1, t2], 1)
# t4 = tf.concat([t1, t2], -1)
# # with tf.control_dependencies([t3, t4]):
# #     train_op = tf.no_op(name='train')
# with tf.Session() as sess:
#     sess.run([t3, t4])
#     print(t3.eval())
#     print(t4.eval())

# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = tf.expand_dims(t1, axis=1)
# with tf.Session() as sess:
#     sess.run(t2)
#     print(t2.eval())
#     print(t2.shape)


# t1 = tf.constant([1,2,3])
# t2 = tf.constant([2,2,2])
# t3 = t1 + t2
# with tf.Session() as sess:
#     sess.run(t3)
#     print(t3.eval())


# 定义一个矩阵a，表示需要被卷积的矩阵。
a = np.array(np.arange(1, 1 + 4).reshape([1, 5, 1]), dtype=np.float32)
print(a)
# 卷积核，此处卷积核的数目为1
kernel = np.array(np.arange(1, 1 + 4), dtype=np.float32).reshape([2, 2, 1])
print(kernel)
# 进行conv1d卷积
conv1d = tf.nn.conv1d(a, kernel, 1, padding='SAME')
# conv1d = tf.nn.conv1d(a, kernel, 1, 'VALID')
with tf.Session() as sess:
    # 初始化
    tf.global_variables_initializer().run()
    # 输出卷积值
    print(sess.run(conv1d))
    print(conv1d.shape)
