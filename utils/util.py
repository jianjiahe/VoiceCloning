import logging
import codecs
import tensorflow as tf
import os
import glob
from matplotlib import pyplot as plt
from hparams import hparams as hp
from utils.text.symbols import Symbols

def list_path(pattern):
    return glob.glob(pattern, recursive=True)


def check_restore_params(saver: tf.train.Saver, sess, run_name, corpus_name=hp.biaobei):
    """
    if checkpoints exits in the give path, then restore the parameters and return True

    :param saver: tf.train.Saver
    :param sess: tf.Session
    :param run_name: directory name of the checkpoints
    :param corpus_name: corpus name
    :return: boolean, return true if checkpoints found else False
    """
    directory = os.path.join(hp.CKP_DIR, corpus_name, run_name, '')
    checkpoint = tf.train.get_checkpoint_state(directory)
    if checkpoint and checkpoint.model_checkpoint_path:
        logging.info('Restoring checkpoints...')
        saver.restore(sess, checkpoint.model_checkpoint_path)
        return True
    else:
        logging.info('No checkpoint found, use initialized parameters.')
        return False


def check_specific_params(saver: tf.train.Saver, sess, path):
    """
    Restore parameters in the specific checkpoint file
    :param saver: tf.train.Saver
    :param sess: tf.Session
    :param path: given checkpoint file path
    :return:
    """
    if os.path.exists(path):
        saver.restore(sess, path)
        return True
    else:
        return False


def text2sequence(text) -> list:
    sequence = [Symbols.sym2id_dict[s] for s in text if s in Symbols.characters]
    sequence.append(Symbols.sym2id_dict[Symbols.eos])
    return sequence

def load_pinyin(datatxt_path: str, data_num=0):
    with codecs.open(datatxt_path) as f:
        line_0 = f.readlines()[data_num]
        return line_0.split('|')[-1]


def plot_alignment(alignment, path, info=None):
    fig, ax = plt.subplots()
    im = ax.imshow(
        alignment,
        aspect='auto',
        origin='lower',
        interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder time step'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder time step')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


if __name__ == '__main__':
    pinyin = load_pinyin_one('/home/the/disk_ssd/disk_ubuntu18.04/tacotron/input/biaobei/train_data.txt')
    print(pinyin)
    print(text2sequence(pinyin))
