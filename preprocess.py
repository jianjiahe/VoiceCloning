import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import thchs30, biaobei
from hparams import hparams as hp
import csv
from utils.spectrogram import get_linear_and_mel


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train_data.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frame = sum([m[3] for m in metadata])
    hours = frame * hp.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frame, hours))
    print('Max input length:  %d' % max(len(m[4]) for m in metadata))
    print('Max output length: %d' % max(m[3] for m in metadata))

def write_metadata_csv(metadata, out_dir):
    with open(os.path.join(out_dir, 'train_data.csv'), 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f, dialect=("excel"))
        csvwriter.writerow([
                 "wav_file",
                 "spec_filepath",
                 "mel_filepath",
                 "n_frame",
                 "pinyin"
        ])
        csvwriter.writerows(metadata)
    #     for m in metadata:
    #         f.write('|'.join([str(x) for x in m]) + '\n')
    # frame = sum([m[3] for m in metadata])
    # hours = frame * hp.frame_shift_ms / (3600 * 1000)
    # print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frame, hours))
    # print('Max input length:  %d' % max(len(m[4]) for m in metadata))
    # print('Max output length: %d' % max(m[3] for m in metadata))

def preprocess_thchs30(args):
    in_dir = os.path.join(args.base_dir, 'thchs30', 'data')
    out_dir = os.path.join(args.out_dir, 'thchs30')
    os.makedirs(out_dir, exist_ok=True)
    metadata = thchs30.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

def preprocess_biaobei(args):
    in_dir = os.path.join(args.base_dir, 'biaobei')
    out_dir = os.path.join(args.out_dir, 'biaobei')
    print(in_dir)
    os.makedirs(out_dir, exist_ok=True)
    metadata = biaobei.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    # write_metadata(metadata, out_dir)
    write_metadata_csv(metadata, out_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('/home/the/Data'))
    parser.add_argument('--out_dir', default='/home/the/disk_ssd/disk_ubuntu18.04/tacotron/input')
    parser.add_argument('--dataset', default='biaobei', choices=['biaobei', 'thchs30'])
    parser.add_argument('--num_workers', type=int, default=cpu_count())

    args = parser.parse_args()
    print(args.base_dir)
    if args.dataset == 'biaobei':
        preprocess_biaobei(args)
    elif args.dataset == 'thchs30':
        preprocess_thchs30(args)


if __name__ == '__main__':
    main()