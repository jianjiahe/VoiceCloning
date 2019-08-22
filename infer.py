import os

from hparams import hparams as hp
from utils.InferBase import InferBase
from utils.audio import inv_preemphasis, save_wav
from utils.spectrogram import griffin_lim
from utils.util import plot_alignment
import numpy as np
import tensorflow as tf
import pinyin

from utils.util import text2sequence

class Infer(InferBase):
    def __init__(self,
                 corpus_name=hp.biaobei,
                 run_name='',
                 mel_filters=hp.num_mels,
                 n_fft=hp.n_fft
                 ):
        super(Infer, self).__init__(corpus_name=corpus_name,
                                    run_name=run_name,
                                    mel_filters=mel_filters,
                                    n_fft=n_fft)

    @staticmethod
    def construct_input(text: str):
        pinyin_ = pinyin.get(text, delimiter=' ', format='numerical')
        print(pinyin_)
        seq = np.asarray(text2sequence(pinyin_))
        length = len(seq)
        print('np.expand_dims(seq, 0).shape', np.expand_dims(seq, 0).shape)
        return np.expand_dims(seq, 0), np.asarray([length])

    def synthesis(self, text, wav_path: str, use_log=True):
        if not self.has_built:
            self._build_graph()
        with tf.Session() as sess:
            self._graph_init(sess)
            net_input, input_length = self.construct_input(text)
            print(input_length)
            print(input_length.shape)
            linear_output, alignments = sess.run([self.linear_output, self.alignments],
                                                 feed_dict={self.inputs: net_input,
                                                            self.input_length: input_length})

            linear_mag = linear_output[0] if not use_log else \
                np.exp(linear_output[0] + hp.linear_log_center)
            print(np.min(linear_output))
            print(np.max(linear_output))
            audio = inv_preemphasis(griffin_lim(np.transpose(linear_mag)))

            if not os.path.exists(os.path.dirname(wav_path)):
                os.makedirs(os.path.dirname(wav_path))
            save_wav(audio, wav_path)
            plot_alignment(alignment=alignments[0], path=wav_path + '.png')


def test():
    infer = Infer(run_name='th30-momentum_l20.001_lr0.01-0.001-rec_loss1e-5-not_accumulate-norm-cbhg_res_gru-log')
    infer.synthesis('绿是阳春烟景，大块文章的底色', wav_path='eval/epoch0.wav')


def test_wav():
    wav_path = 'data/wav/test.wav'
    linear_mag = np.load('out/th30/linear_spec/A2_0-linear.npy')

    audio = inv_preemphasis(griffin_lim(linear_mag))
    # audio = griffin_lim(linear_mag)
    if not os.path.exists(os.path.dirname(wav_path)):
        os.makedirs(os.path.dirname(wav_path))
    save_wav(audio, wav_path)


def test_input():
    txt = '绿是阳春烟景，大块文章的底色'
    inputs, length = Infer().construct_input(txt)
    print(inputs)


if __name__ == '__main__':
    # test_wav()
    test_input()