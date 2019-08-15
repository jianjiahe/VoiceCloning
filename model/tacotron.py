from hparams import hparams as hp
from util.text.symbols import Symbols
import tensorflow as tf
from model.util.basis import pre_net, cbhg
from model.util.attention import Attention
from model.util.rnn_wrapper import DecoderWrapper
from model.util.custum_decoder import CustomDecoder
from model.util.helper import TacoTestHelper, TacoTrainHelper
from tensorflow.contrib.rnn import MultiRNNCell,ResidualWrapper, GRUCell
from tensorflow.contrib.seq2seq import dynamic_decode

class Tacotron:
    def __init__(self, training=False):
        self._training = training
        self._embedding_depth = hp.embed_depth
        self._attention_depth = hp.attention_depth
        self._mel_filters = hp.num_mels
        self._output_per_size = hp.outputs_per_step
        self._max_iterations = hp.max_iters
        self._linear_size = hp.num_freq

    @property
    def _multi_rnn_cell(self):
        return MultiRNNCell([
            ResidualWrapper(GRUCell(hp.decoder_depth)),
            ResidualWrapper(GRUCell(hp.decoder_depth))
        ], state_is_tuple=True)

    def get_helper(self, inputs, mel_targets, global_step, batch_size):
        if self._training:
            return TacoTrainHelper(inputs, mel_targets,
                                   output_dim=self._mel_filters,
                                   r=self._output_per_step,
                                   global_step=global_step)
        else:
            return TacoTestHelper(batch_size=batch_size,
                                  output_dim=self._mel_filters,
                                  r=self._output_per_step)

    def embedding_layer(self, inputs):
        with tf.variable_scope('EmbeddingLayer'):
            embedding_table = tf.get_variable('embedding', [len(Symbols.symbols), self._embedding_depth],
                                              initializer=tf.truncated_normal_initializer(stddev=0.5))
            return tf.nn.embedding_lookup(embedding_table, inputs)

    def encoder(self, embedding_outputs, input_length, training=False):
        with tf.variable_scope('Encoder'):
            out = pre_net(embedding_outputs, training=training, scope='Encoder_pre_net', use_bn=True)  # why use_bn=True
            out = cbhg(out, input_length, training, [128, hp.prenet_depths[-1]], k=16, depth=128, scope='Encoder_CBHG')
            return out

    def decoder(self, attention_mechanism, batch_size, inputs, mel_targets, global_step):
        with tf.variable_scope('Decoder'):
            decoder_cell = DecoderWrapper(self._training, attention_mechanism,
                                          self._multi_rnn_cell, r=self._output_per_size)

            decoder_init_state = decoder_cell.zero_state(batch_size=batch_size, dtype=tf.float32)

            helper = self.get_helper(inputs, mel_targets, global_step, batch_size)

            outs = dynamic_decode(
                CustomDecoder(decoder_cell, helper, decoder_init_state), maximum_iterations=self._max_iterations)

            return outs

    def refine_linear(self, linear):
        with tf.variable_scope('LinearOutput'):
            linear_outputs = tf.layers.dense(linear, self._linear_size, use_bias=True)
            return linear_outputs

    def infer(self, inputs, input_length, mel_targets=None, global_step=None):
        batch_size = tf.shape(inputs[0])

        embedding_inputs = self.embedding_layer(inputs)

        encoder_outputs = self.encoder(embedding_inputs, input_length, training=self._training)

        attention_mechanism = Attention(self._attention_depth, encoder_outputs)

        (decoder_outputs, stop_token_outputs, _), final_decoder_state, _ =\
            self.decoder(attention_mechanism, batch_size, inputs, mel_targets, global_step)

        with tf.variable_scope('MelOutput'):
            mel_outputs = tf.reshape(decoder_outputs, [batch_size, -1, self._mel_filters])

        with tf.variable_scope('StopTokenOutput'):
            stop_token_outputs = tf.reshape(stop_token_outputs, [batch_size, -1])

        # post processing cbhg
        _, post_outputs = cbhg(mel_outputs, None, self._training, projection=[256, 64], k=8,
                               depth=hp.postnet_depth, scope='Post-processing-CBHG')

        linear_outputs = self.refine_linear(post_outputs)
        with tf.variable_scope('Alignments'):
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

        if self._training:
            # summaries
            with tf.variable_scope('intermediate'):
                tf.summary.histogram('Embedding', embedding_inputs)
                tf.summary.histogram('Encoder_outputs', encoder_outputs)
                tf.summary.histogram('decoder_outputs', decoder_outputs)
            with tf.variable_scope('out'):
                tf.summary.histogram('stop_token_outputs', stop_token_outputs)
                tf.summary.histogram('mel_output', mel_outputs)
                tf.summary.histogram('linear_output', linear_outputs)
                tf.summary.image('alignments', tf.expand_dims(alignments, -1))
                tf.summary.histogram('alignments', alignments)

        return mel_outputs, linear_outputs, stop_token_outputs, alignments