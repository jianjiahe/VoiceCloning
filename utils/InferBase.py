import os
import numpy as np
import tensorflow as tf
from hparams import hparams as hp
from utils.util import check_restore_params
from models.tacotron import Tacotron


class InferBase:

    def __init__(self,
                 corpus_name=hp.th30,
                 run_name='',
                 mel_filters=hp.num_mels,
                 n_fft=hp.n_fft):

        self.corpus_name = corpus_name
        self.run_name = run_name
        self.mel_filters = mel_filters
        self.n_fft = n_fft
        self.has_built = False

    def _build_graph(self):
        tf.reset_default_graph()
        with tf.variable_scope('evaluation_data'):
            self.inputs = tf.placeholder(tf.int32, shape=[None, None])
            self.input_length = tf.placeholder(tf.int32, shape=[None])

        self.mel_outputs, self.linear_output, self.stop_token_output, self.alignments \
            = Tacotron(training=False).infer(self.inputs, input_length=self.input_length)

        self.writer = tf.summary.FileWriter(os.path.join(hp.CKP_DIR, self.corpus_name, self.run_name))

        self.has_built = True

    @staticmethod
    def finalize():
        tf.get_default_graph().finalize()

    def _graph_init(self, sess):
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        check_restore_params(saver, sess, self.run_name, corpus_name=self.corpus_name)

    def infer_batch(self, sess: tf.Session, batch_input, batch_input_length) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        with the sess as the context, infer the given batch input
        :param sess:
        :param batch_input:
        :param batch_input_length:
        :return:
        """
        mel_output, linear_output, stop_token_output = \
            sess.run([self.mel_outputs, self.linear_output, self.stop_token_output],
                     feed_dict={self.inputs: batch_input, self.input_length: batch_input_length})
        return mel_output, linear_output, stop_token_output
