import librosa
import librosa.filters
import math
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
from hparams import hparams as hp


def save_wav(audio: np.ndarray, out_path):
    sf.write(out_path, audio, hp.sample_rate, subtype='PCM_16')


def load_wav(path):
    return librosa.core.load(path, sr=hp.sample_rate)[0]


def trim_silence(wav):
    return librosa.effects.trim(wav, top_db=60, frame_length=512, hop_length=128)[0]


def preemphasis(x):
    return signal.lfilter([1, -hp.preemphasis], [1], x)


def inv_preemphasis(x):
    return signal.lfilter([1], [1, -hp.preemphasis], x)


def _stft_parameters():
    n_fft = hp.n_fft
    hop_length = int(hp.frame_shift_ms / 1000 * hp.sample_rate)
    win_length = int(hp.frame_length_ms / 1000 * hp.sample_rate)
    return n_fft, hop_length, win_length


def stft(y):
    n_fft, hop_length, win_length = _stft_parameters()
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)


def _amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))
    #   return 20 * np.log10(x)


def _normalize(S):
    # symmetric mels
    return 2 * hp.max_abs_value * ((S - hp.min_level_db) / -hp.min_level_db) - hp.max_abs_value


def spectrogram(stft_D):
    S = _amp_to_db(np.abs(stft_D)) - hp.ref_level_db
    return _normalize(S)


# mel
_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    n_fft = hp.n_fft
    assert hp.fmax < hp.sample_rate // 2
    return librosa.filters.mel(hp.sample_rate, n_fft, n_mels=hp.num_mels, fmin=hp.fmin, fmax=hp.fmax)  # wrong


def melspectrogram(stft_D):
    S = _amp_to_db(_linear_to_mel(np.abs(stft_D))) - hp.ref_level_db  # wrong
    return _normalize(S)
