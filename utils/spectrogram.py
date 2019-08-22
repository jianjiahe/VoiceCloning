import librosa
from python_speech_features import fbank
from utils import audio
import hparams
from hparams import hparams as hp
import logging
import numpy as np
from matplotlib import pyplot as plt

def get_linear_and_mel(wav: np.ndarray):
    # pre emphasis
    wav = audio.preemphasis(wav)
    linear_spec = np.absolute(librosa.stft(wav, n_fft=hp.n_fft, hop_length=hp.frame_shift_samples,
                                           win_length=hp.frame_length_samples,
                                           window=hp.window))  # shape: (1+n_fft/2, ts) = (257, ts)
    # power_spec_in_db = 40 * np.log10(np.maximum(1e-5, spec))  # multiply by 40 to get the power spectrogram in db`
    mel_basic = librosa.filters.mel(hp.sample_rate, hp.n_fft, n_mels=hp.num_mels, norm=None)  # (64, 257)
    # mel_spec = np.dot(mel_basic, np.power(linear_spec, 2))
    mel_spec = np.dot(mel_basic, linear_spec) # (64, ts)

    return linear_spec, mel_spec  # linear_spec, mel_spec without log


def griffin_lim(linear_spec):
    angles = np.exp(2j * np.pi * np.random.rand(*linear_spec.shape))
    audio = librosa.istft(linear_spec * angles, hop_length=hp.frame_shift_samples,
                          win_length=hp.frame_length_samples, window=hp.window)
    for i in range(hp.griffin_lim_iterations):
        complex_spec = librosa.stft(audio, n_fft=hp.n_fft, hop_length=hp.frame_shift_samples,
                                    win_length=hp.frame_length_samples, window=hp.window)
        angles = np.exp(1j * np.angle(complex_spec))
        wav = librosa.istft(linear_spec * angles, hop_length=hp.frame_shift_samples, win_length=hp.frame_length_samples,
                            window=hp.window)
    return wav


def linear2wav(linear_spec):
    wav = griffin_lim(linear_spec)
    return audio.inv_preemphasis(wav)


def test():
    wav_path = '/home/the/Data/biaobei/000001.wav'
    wav = audio.load_wav(wav_path)
    logging.info('audio.shape: {0}'.format(wav.shape))
    linear, mel = get_linear_and_mel(wav)
    plt.figure()
    plt.imshow(np.log(linear))
    plt.figure()
    # plt.imshow(np.log(mel[:, :200]))
    plt.imshow(np.log(mel))
    plt.show()


if __name__ == '__main__':
    hparams.basic_config()
    test()
