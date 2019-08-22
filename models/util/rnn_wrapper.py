import tensorflow as tf
from hparams import hparams as hp
from models.util.basis import pre_net

from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.rnn import RNNCell

import collections


def compute_attention(attention_mechanism, cell_output, attention_state, attention_layer):
    """

        :param attention_mechanism: given attention mechanism
        :param cell_output: rnn cell output, that is query
        :param attention_state: previous alignments
        :param attention_layer:

        :return: a tuple : (attention output, alignments, accumulative alignments(useless))
        """
    alignments, accumulated_alignments = attention_mechanism(cell_output, state=attention_state)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments, accumulated_alignments


class TacotronDecoderCellState(
    collections.namedtuple("TacotronDecoderCellState",
                           ("cell_state", "attention", "time", "alignments", "alignment_history"))):
    """
    - `cell_state`: The state of the wrapped `RNNCell` at the previous time
    step.
    - `attention`: The attention emitted at the previous time step.
    - `time`: int32 scalar containing the current time step.
    - `alignments`: A single or tuple of `Tensor`(s) containing the alignments
     emitted at the previous time step for each attention mechanism.
    - `alignment_history`: a single or tuple of `TensorArray`(s)
     containing alignment matrices from all time steps for each attention
     mechanism. Call `stack()` on each to convert to a `Tensor`.
  """

    def replace(self, **kwargs):
        """
        Clones the current state while overwriting components provided by kwargs.
        """
        return super(TacotronDecoderCellState, self)._replace(**kwargs)


class DecoderWrapper(RNNCell):

    def __init__(self, training, attention_mechanism, rnn_cell, r=hp.outputs_per_step):
        super(DecoderWrapper, self).__init__()
        self._training = training
        self._attention_mechanism = attention_mechanism
        self._cell = rnn_cell
        self._attention_layer_size = self._attention_mechanism.values.get_shape()[-1].value
        self._reduction_rate = r

    @property
    def state_size(self):
        return TacotronDecoderCellState(
            cell_state=self._cell.state_size,
            time=tf.TensorShape([]),
            attention=self._attention_layer_size,
            alignments=self._attention_mechanism.alignments_size,
            alignment_history=())

    @property
    def output_size(self):
        return hp.num_mels * self._reduction_rate

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size, self._attention_mechanism.batch_size, message=error_message)]

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of TacotronDecoderCell %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                    lambda s: tf.identity(s, name="checked_cell_state"),
                    cell_state)
            return TacotronDecoderCellState(
                cell_state=cell_state,
                time=tf.zeros([], dtype=tf.int32),
                attention=rnn_cell_impl._zero_state_tensors(self._attention_layer_size, batch_size, dtype),
                alignments=self._attention_mechanism.initial_alignments(batch_size, dtype),
                alignment_history=tf.TensorArray(dtype=dtype, size=0, dynamic_size=True))

    def __call__(self, inputs, state, scope=None):
        """

        :param inputs: inputs, last out
        :param state: last generated state
        :param scope:
        :return:
        """
        pre_net_out = pre_net(inputs, self._training, scope='decoder-pre-net', use_bn=False)

        rnn_input = tf.concat([pre_net_out, state.attention], axis=-1)

        rnn_out, cur_cell_state = self._cell(
            tf.layers.dense(rnn_input, hp.decoder_depth), state.cell_state
        )

        previous_alignments = state.alignments
        previous_alignment_history = state.alignment_history

        context_vector, alignments, accumulated_alignments = compute_attention(self._attention_mechanism,
                                                                               rnn_out,
                                                                               previous_alignments,
                                                                               attention_layer=None)
        projections_input = tf.concat([rnn_out, context_vector], axis=-1)  # (bs, ?, rnn + context)
        cell_output = tf.layers.dense(projections_input, hp.num_mels * self._reduction_rate,
                                      name='frame_projection', use_bias=True)  # (bs, ?, mel * r)

        stop_tokens = tf.layers.dense(projections_input, self._reduction_rate, activation=None,
                                      name='stop_token', use_bias=True)  # (bs, ? r)
        if not self._training:  # when training, sigmoid is integrated in the sigmoid cross entropy loss
            stop_tokens = tf.nn.sigmoid(stop_tokens)

        alignment_history = previous_alignment_history.write(state.time, alignments)

        # prepare next decoder state
        cur_state = TacotronDecoderCellState(
            time=state.time + 1,
            cell_state=cur_cell_state,
            attention=context_vector,
            alignments=alignments,
            alignment_history=alignment_history)

        return (cell_output, stop_tokens), cur_state
