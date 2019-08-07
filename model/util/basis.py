import tensorflow as tf
from tensorflow import layers
from tensorflow.contrib.rnn import ResidualWrapper
from hparams import hparams as hp


def pre_net(inputs, training=False, scope='pre_net', use_bn=False):
    out = inputs if not use_bn else tf.layers.batch_normalization(inputs, axis=-1, training=training)
    layers_sizes = hp.prenet_depths
    with tf.variable_scope(scope):
        for size in layers_sizes:
            out = tf.layers.dense(
                out,
                units=size,
                use_bias=not use_bn
            )
            if use_bn:
                out = tf.layers.batch_normalization(out, axis=-1, training=training)
            out = tf.nn.relu(out)
            out = tf.layers.dropout(out, rate=0.5, training=training)
    return out


def cbhg(inputs, input_length, training, projection: list, k, depth=128, scope='CBHG'):
    with tf.variable_scope(scope):
        with tf.variable_scope('conv1d_bank_stacking'):
            conv1d_banks = []
            for i in range(1, k + 1):
                conv_out = layers.conv1d(inputs, 128, i, padding='same', use_bias=False)
                bn_out = tf.layers.batch_normalization(conv_out, axis=-1, training=training)
                conv1d_banks.append(bn_out)
            # stacking
            out = tf.concat(conv1d_banks, axis=-1)  # 沿着最后一维链接起来，即有k+1个一维卷积的结果在一个list里面，即
            out = tf.nn.relu(out)
        tf.summary.histogram('conv1d_bank_stacking', out)

        with tf.variable_scope('max_pooling'):
            out = tf.layers.max_pooling1d(out, 2, 1, padding='same')
        tf.summary.histogram('max_pooling', out)

        with tf.variable_scope('con1d_projection'):
            projector1_out = layers.conv1d(out, projection[0], 3, padding='same', use_bias=False)
            out = tf.layers.batch_normalization(projector1_out, axis=-1, training=training)
            out = tf.nn.relu(out)
            projector2_out = layers.conv1d(out, projection[1], 3, padding='same', use_bias=False)
        tf.summary.histogram('conv1d_projection', projector2_out)

        # residual connection
        residual_out = inputs + projector2_out
        tf.summary.histogram('conv1d_projection', residual_out)
        out = residual_out

        if out.get_shape()[2] != depth:
            out = layers.dense(out, depth, use_bias=False)
        for i in range(4):
            out = highwaynet(out, 'highway_%d' % (i + 1), depth)
        tf.summary.histogram('highway_out', out)

        with tf.variable_scope('Bidirectional_GRU'):
            outputs, states = tf.nn.bidirectional_dynamic_rnn(
                tf.nn.rnn_cell.GRUCell(depth),
                tf.nn.rnn_cell.GRUCell(depth),
                out,
                sequence_length=input_length,
                dtype=tf.float32
            )
        out = tf.concat(outputs, axis=2)
        return residual_out, tf.layers.batch_normalization(out, axis=-1, training=training)

    pass


def highwaynet(inputs, scope, depth):
    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=depth,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)
