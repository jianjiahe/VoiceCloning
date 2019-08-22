from tensorflow.contrib.seq2seq import Helper
import tensorflow as tf


def go_frames(batch_size, output_dim):
    return tf.tile([[0.]], [batch_size, output_dim])


class TacoTestHelper(Helper):
    def __init__(self, batch_size, output_dim, r):
        with tf.name_scope('TacoHelper'):
            self._batch_size = batch_size
            self._output_dim = output_dim
            self._reduction_factor = r

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def token_output_size(self):
        return self._reduction_factor

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        return tf.tile([False], [self.batch_size]), go_frames(self.batch_size, self._output_dim)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self.batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
        with tf.variable_scope('TacoTestHelper'):
            finished = tf.reduce_any(tf.cast(tf.round(stop_token_preds), tf.bool))

            # Feed last output frame as next input. outputs is [N, output_dim * r]
            next_inputs = outputs[:, -self._output_dim:]
            return finished, next_inputs, state


class TacoTrainHelper(Helper):
    def __init__(self, inputs, targets, output_dim, r, global_step):
        self._batch_size = tf.shape(inputs)[0]
        self._output_dim = output_dim
        self._reduction_factor = r
        self._ratio = tf.convert_to_tensor(1.)
        self._global_step = global_step

        self._targets = targets[:, r - 1::r, :]  # num_steps = tf.shape(self._targets)[1]

        self._lengths = tf.tile([tf.shape(self._targets)[1]], [self._batch_size])

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def token_output_size(self):
        return self._reduction_factor

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def initialize(self, name=None):
        self._ratio = teacher_forcing_ratio_decay(1., self._global_step)
        return tf.tile([False], [self.batch_size]), go_frames(self.batch_size, self._output_dim)

    def sample(self, time, outputs, state, name=None):
        return tf.tile([0], [self.batch_size])

    def next_inputs(self, time, outputs, state, sample_ids, stop_token_preds, name=None):
        with tf.variable_scope(name, 'TacoTrainHelper'):

            finished = (time + 1 >= self._lengths)
            next_inputs = tf.cond(tf.less(tf.random_uniform([], minval=0, maxval=1, dtype=tf.float32), self._ratio),
                                  lambda: self._targets[:, time, :],  # return the true frame
                                  lambda: outputs[:, -self._output_dim:])
            return finished, next_inputs, state


def teacher_forcing_ratio_decay(init_tfr, global_step, init_steps=10000):
    """

    :param init_steps: step number for initialize
    :param init_tfr: initial value
    :param global_step: global step in the graph
    :return:
    """
    tfr = tf.train.cosine_decay(init_tfr,
                                global_step=global_step - init_steps,
                                decay_steps=150000,
                                alpha=0.,
                                name='tfr_cosine_decay')
    narrow_tfr = tf.cond(
        tf.less(global_step, init_steps),
        lambda: tf.convert_to_tensor(init_tfr),
        lambda: tfr,
    )
    tf.summary.scalar('tfr_ratio', narrow_tfr)
    return narrow_tfr
