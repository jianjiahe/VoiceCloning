import tensorflow as tf
from tensorflow.contrib.seq2seq import BahdanauAttention
from hparams import hparams as hp


class Attention(BahdanauAttention):
    def __init__(self,
                 num_units,
                 memory,
                 l2=hp.l2,
                 name='attention'):
        super(Attention, self).__init__(
            num_units=num_units,
            memory=memory,
            memory_sequence_length=None,
            probability_fn=None,
            name='attention'
        )
        self.l2=l2

    def __call__(self, query, state):
        """

        :param query: shape: (batch_size, query_depth)
        :param state: shape: (batch_size, alignment_size) | previous alignments, where alignment_size is max_time
        :return: alignment, accumulated alignment
        """
        previous_alignment = state
        with tf.variable_scope(None, 'attention', [query]):
            processed_query = self.query_layer(query) if self.query_layer else query
            processed_query = tf.expand_dims(processed_query, axis=1)  # -> (batch_size, 1 , num_units)

            expanded_alignments = tf.expand_dims(previous_alignment, axis=2)  # -> (batch_size, max_time, 1)
            f = tf.layers.conv1d(expanded_alignments, filters=32, kernel_size=31, padding='same', use_bias=True)  # -> (batch_size, max_time, 32)
            processed_location_features = tf.layers.dense(f, units=self._num_units, use_bias=False) # -> (batch_size, max_time, num_units)

            score = self._location_sensitive_score(processed_query, processed_location_features, self.keys) # (batch_size, max_time)

            alignments = self._probability_fn(score, previous_alignment)
            accumulated = alignments + previous_alignment
            return alignments, accumulated

    @staticmethod
    def _location_sensitive_score(w_query, w_location, w_keys):
        data_type = w_query.dtype
        num_units = w_keys.get_shape()[-1]

        v_a = tf.get_variable('attention_variable', shape=[num_units], dtype=data_type)
        b_a = tf.get_variable('attention_bias', shape=[num_units], dtype=data_type, initializer=tf.zeros_initializer())

        score = tf.reduce_sum(v_a * tf.tanh(w_keys + w_query + w_location + b_a), axis=2)

        return score


