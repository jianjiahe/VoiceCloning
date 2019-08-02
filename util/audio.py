import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from hparams import hparams

def load_wav(path):
    return librosa.core.load(path, sr=hparams.sample_rate)[0]

def trim_silence(wav):
    return librosa.effects.trim(wav, top_db= 60, frame_length=512, hop_length=128)[0]

def preemphasis(x):
    return signal.lfilter([1, -hparams.preemphasis], [1], x)

''' 
    num_mels=80,
    num_freq=2049,
    sample_rate=48000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,
    max_frame_num=1000,
    max_abs_value = 4,
    fmin = 125, # for male, set 55
    fmax = 7600, # for male, set 3600
'''

def _stft_parameters():
    n_fft = (hparams.num_freq - 1) * 2
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    win_length = int(hparams.frame_length_ms / 1000 * hparams.sample_rate)
    return n_fft, hop_length, win_length

def stft(y):
  n_fft, hop_length, win_length = _stft_parameters()
  return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)

def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))
    #   return 20 * np.log10(x)


def _normalize(S):
    # symmetric mels
    return 2 * hparams.max_abs_value * ((S - hparams.min_level_db) / -hparams.min_level_db) - hparams.max_abs_value

def spectrogram(stft_D):
    S = _amp_to_db(np.abs(stft_D)) - hparams.ref_level_db
    return _normalize(S)

# mel
_mel_basis = None

def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)

def _build_mel_basis():
    n_fft = (hparams.num_freq - 1) * 2
    assert hparams.fmax < hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, n_fft, n_mels=hparams.num_mels, fmin=hparams.fmin, fmax=hparams.fmax) #wrong

def melspectrogram(stft_D):
    S = _amp_to_db(_linear_to_mel(np.abs(stft_D))) - hparams.ref_level_db # wrong
    return _normalize(S)