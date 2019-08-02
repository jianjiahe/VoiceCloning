import argparse
import os
from multiprocessing import cpu_count
from tqdm import tqdm
from datasets import thchs30
from hparams import hparams as hp

def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train_data.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    frame = sum([m[3] for m in metadata])
    hours = frame * hp.frame_shift_ms / (3600 * 1000)
    print('Wrote %d utterances, %d frames (%.2f hours)' % (len(metadata), frame, hours))
    print('Max input length:  %d' % max(len(m[4]) for m in metadata))
    print('Max output length: %d' % max(m[3] for m in metadata))



def preprocess_thchs30(args):
    in_dir = os.path.join(args.base_dir, 'thchs30', 'data')
    out_dir = os.path.join(args.out_dir, 'thchs30')
    os.makedirs(out_dir, exist_ok=True)
    metadata = thchs30.build_from_path(in_dir, out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

def preprocess_biaobei(args):
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default=os.path.expanduser('/home/the/Data'))
    parser.add_argument('--out_dir', default='/home/the/disk_ssd/disk_ubuntu18.04/tacotron/input/thchs30')
    parser.add_argument('--dataset', default='thchs30', choices=['biaobei', 'thchs30'])
    parser.add_argument('--num_workers', type=int, default=cpu_count())

    args = parser.parse_args()
    print(args.base_dir)
    if args.dataset == 'biaobei':
        preprocess_biaobei(args)
    elif args.dataset == 'thchs30':
        preprocess_thchs30(args)


if __name__ == '__main__':
    main()