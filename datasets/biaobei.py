from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
import glob
from utils import audio
from hparams import hparams as hp
from utils.spectrogram import get_linear_and_mel
import librosa

def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 0
    # trn_files = glob.glob(os.path.join(in_dir, '*.wave'))
    # for trn in trn_files:
    #     with open(trn) as f:
    #         pinyin = f.readlines()[1].strip('\n')
    #         wav_file = trn[:-4]
    #         task = partial(_process_utterance, out_dir, index, wav_file, pinyin)
    #         index += 1
    #         futures.append(executor.submit(task))
    with open(os.path.join(in_dir, '000001-010000.txt'), 'r') as f:
        lines = f.readlines()
        for line_num, line in enumerate(lines):
            if line_num & 1 == 0:
                wav_file = line.split('\t')[0] + '.wav'
                wav_file = os.path.join(in_dir, wav_file)
                # print(line_num)
            else:
                pinyin = line.strip()
                index += 1
                task = partial(_process_utterance, out_dir, index, wav_file, pinyin)
                futures.append(executor.submit(task))

    return [future.result() for future in tqdm(futures) if future.result() is not None]

def _process_utterance(out_dir, index, wav_file, pinyin):
    wav = librosa.load(wav_file, sr=hp.sample_rate)[0]

    # rescale wav for unified volume
    wav = wav / np.abs(wav).max() * 0.999

    # trim silence frame, [ref=np.mean top_db=13]对应ratio0.05  -> 8(0.15), 如果是np.max则对应top_db=17
    wav = librosa.effects.trim(wav, ref=np.mean, top_db=8, frame_length=2048, hop_length=512)[0]

    spectrogram, mel_spectrogram = get_linear_and_mel(wav)
    n_frame = spectrogram.shape[1]
    if n_frame > hp.max_frame_num:
        return None

    spec_filename = 'spec-%05d.npy' % index
    mel_filename = 'mel-%05d.npy' % index
    spec_dir = os.path.join(out_dir, 'spectrogram')
    mel_dir = os.path.join(out_dir, 'mel_spectrogram')
    spec_filepath = os.path.join(spec_dir, spec_filename)
    mel_filepath = os.path.join(mel_dir, mel_filename)
    os.makedirs(spec_dir, exist_ok=True)
    os.makedirs(mel_dir, exist_ok=True)
    np.save(spec_filepath, spectrogram, allow_pickle=False)
    np.save(mel_filepath, mel_spectrogram, allow_pickle=False)

    return (wav_file, spec_filepath, mel_filepath, n_frame, pinyin)

# def _process_utterance(out_dir, index, wav_file, pinyin):
#     wav = audio.load_wav(wav_file)
#     wav = wav / np.abs(wav).max() * 0.999
#     wav = audio.trim_silence(wav)
#
#     stft_D = audio.stft(audio.preemphasis(wav))
#     spectrogram = audio.spectrogram(stft_D).astype(np.float32)
#
#     n_frame = spectrogram.shape[1]
#     if n_frame > hp.max_frame_num:
#         return None
#     mel_spectrogram = audio.melspectrogram(stft_D).astype(np.float32)
#
#     spec_filename = 'spec-%05d.npy' % index
#     mel_filename = 'mel-%05d.npy' % index
#     spec_dir = os.path.join(out_dir, 'spectrogram')
#     mel_dir = os.path.join(out_dir, 'mel_spectrogram')
#     spec_filepath = os.path.join(spec_dir, spec_filename)
#     mel_filepath = os.path.join(mel_dir, mel_filename)
#     os.makedirs(spec_dir, exist_ok=True)
#     os.makedirs(mel_dir, exist_ok=True)
#     np.save(spec_filepath, spectrogram.T, allow_pickle=False)
#     np.save(mel_filepath, mel_spectrogram.T, allow_pickle=False)
#
#     return (wav_file, spec_filepath, mel_filepath, n_frame, pinyin)