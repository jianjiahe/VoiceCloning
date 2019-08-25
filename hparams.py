import tensorflow as tf
import logging
import pandas as pd
import numpy as np
import sys

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    cleaners='basic_cleaners',

    # Audio:
    num_mels=64,
    num_freq=257,
    n_fft=512,    # num_freq = n_fft // 2 + 1
    sample_rate=16000,
    frame_length_ms=25,
    frame_shift_ms=10,
    frame_length_samples=400,   #int(sample_rate * frame_length_ms * 0.01),
    frame_shift_samples=160,    #int(sample_rate * frame_shift_ms * 0.01),
    window=np.hamming,
    mel_log_center=-2.,
    linear_log_center=-3.5,
    max_duration_in_sec=12,

    # num_mels=80,
    # num_freq=2049,
    # sample_rate=48000,
    # frame_length_ms=50,
    # frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    max_frame_num=1000,
    max_abs_value=4,
    fmin=125,  # for male, set 55
    fmax=7600,  # for male, set 3600

    # Model:
    # outputs_per_step=5,
    # embed_depth=512,
    # prenet_depths=[256, 256],
    # encoder_depth=256,
    # postnet_depth=512,
    # attention_depth=128,
    # decoder_depth=1024,
    # l2=0.001,
    outputs_per_step=3,
    embed_depth=512,
    prenet_depths=[256, 256],
    encoder_depth=256,
    postnet_depth=256,
    attention_depth=256,
    decoder_depth=1024,
    l2=0.001,

    # Training:
    batch_size=16,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.001,
    decay_learning_rate=True,
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Eval:
    max_iters=300,
    griffin_lim_iters=60,
    power=1.2,  # Power to raise magnitudes to prior to Griffin-Lim

    # CorpusName
    CKP_DIR='checkpoint',
    th30='th30',
    biaobei='biaobei',

    # OptimizerName
    Momentum='momentum',
    Adam = 'adam',


)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

def basic_config():
    logging.basicConfig(handlers=[logging.StreamHandler(stream=sys.stdout)], level=logging.INFO,
                        format='%(asctime)-15s [%(levelname)s] %(filename)s/%(funcName)s | %(message)s')

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option('max_colwidth', 100)
    np.set_printoptions(precision=3, edgeitems=8, suppress=True, linewidth=1000)

# def main():
#   test = hparams_debug_string()
#   print(test)

# if __name__ == "__main__":
#   main()
