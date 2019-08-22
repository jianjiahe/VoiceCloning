import collections
import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.seq2seq import Decoder, Helper


class CustomDecoderOutput(collections.namedtuple('CustomDecoderOutput',
                                                 ('rnn_output', 'token_output', 'sample_id'))):
    pass


class CustomDecoder(Decoder):
    def __init__(self, cell, helper, initial_state, output_layer=None):
        self._cell = cell
        self._helper = helper
        self._initial_state = initial_state
        self._output_layer = output_layer
        self.assert_instance()

    def assert_instance(self):
        rnn_cell_impl.assert_like_rnncell(type(self._cell), self._cell)
        if not isinstance(self._helper, Helper):
            raise TypeError('helper must be a instance of Helper, received : {0}'.format(type(self._helper)))
        if self._output_layer is not None and not isinstance(self._output_layer, tf.layers.Layer):
            raise TypeError('output layer must be a Layer, received: {0}'.format(type(self._output_layer)))

    @property
    def batch_size(self):
        return self._helper.batch_size

    def _rnn_output_size(self):
        size = self._cell.output_size
        if self._output_layer is None:
            return size
        else:
            output_shape_with_unknown_size = nest.map_structure(
                lambda x: tf.TensorShape([None]).concatenate(x), size
            )
            layer_output_shape = self._output_layer._compute_output_shape(output_shape_with_unknown_size)
            return nest.map_structure(lambda x: x[1:], layer_output_shape)

    @property
    def output_size(self):
        return CustomDecoderOutput(rnn_output=self._rnn_output_size(),
                                   token_output=self._helper.token_output_size,
                                   sample_id=self._helper.sample_ids_shape)

    @property
    def output_dtype(self):
        data_type = nest.flatten(self._initial_state)[0].dtype
        return CustomDecoderOutput(nest.map_structure(lambda _: data_type, self._rnn_output_size()),
                                   tf.float32,
                                   self._helper.sample_ids_dtype)


    def initialize(self, name=None):
        return self._helper.initialize() + (self._initial_state,)

    def step(self, time, inputs, state, name=None):
        """

        :param time: time step
        :param inputs: decoder inputs
        :param state: decoder cell state
        :param name:
        :return:
        """
        with tf.variable_scope(name, 'CostumDecoderStep', (time, inputs, state)):
            (cell_outputs, stop_token), cell_state = self._cell(inputs, state)

            if self._output_layer is not None:
                cell_outputs = self._output_layer(cell_outputs)

            sample_ids = self._helper.sample(time=time, outputs=cell_outputs, state=cell_outputs)

            (finished, next_inputs, next_state) = self._helper.next_inputs(
                time=time, outputs=cell_outputs, state=cell_state,
                sample_ids=sample_ids, stop_token_preds=stop_token)

            outputs = CustomDecoderOutput(cell_outputs, stop_token, sample_ids)

            return outputs, next_state, next_inputs, finished


