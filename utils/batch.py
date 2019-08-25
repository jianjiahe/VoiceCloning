import logging

import pandas as pd
import numpy as np
from functools import partial
# from concurrent.futures import ProcessPoolExecutor
from utils.util import text2sequence, load_pinyin
from hparams import hparams as hp
import os

DATAPATH = '/home/the/disk_ssd/disk_ubuntu18.04/tacotron/input'
TARINFILE = 'train_data.csv'

class MiniBatch:

    def __init__(self,
                 corpus_name=hp.biaobei,
                 batch_size=32,
                 max_duration_in_sec=hp.max_duration_in_sec
                 ):
        self.corpus_name = corpus_name
        self.batch_size = batch_size
        self.max_duration_in_sec = max_duration_in_sec
        # self.all_data = pd.read_csv(os.path.join(DATAPATH, corpus_name, TARINFILE))
        self.train_data = pd.read_csv(os.path.join(DATAPATH, corpus_name, TARINFILE))
        self.valid_train_data = self.train_data[self.train_data['n_frame'] <= (12 * 1000 / hp.frame_shift_ms)]
        self.describe()

        # self.executor = ProcessPoolExecutor(max_workers=10)
    def describe(self):
        logging.info('{0} items in total'.format(len(self.train_data)))
        logging.info('{0} valid items in total.'.format(len(self.valid_train_data)))

    def next_batch(self):

        batch_pieces = self.valid_train_data.sample(self.batch_size, replace=False)

        # tasks = []
        # for _, item in batch_pieces.iterrows():
        #     # notation: 调用进程池进行并行处理输入数据, 计算较小时采用进程池效果更慢
        #     task = partial(self.load_one, item)
        #     tasks.append(self.executor.submit(task))
        #
        # batch = [t.result() for t in tasks]

        batch = [self.load_one(item) for _, item in batch_pieces.iterrows()]
        max_input_length = max(x[1] for x in batch)
        max_target_length = max(x[5] for x in batch)   # 要对齐到reduction_rate的倍数
        max_target_length = (max_target_length // hp.outputs_per_step + 1) * hp.outputs_per_step

        # right padding,
        inputs = np.stack([np.pad(x[0], (0, max_input_length - x[1]), mode='constant') for x in batch])
        input_length = np.stack([x[1] for x in batch])
        mel_target = np.transpose(np.stack(      # (-1, 64, -1) -> (-1, -1, 64)
            [np.pad(x[2], ((0, 0), (0, max_target_length - x[5])), mode='constant') for x in batch]),
            axes=[0, 2, 1]
        )
        linear_target = np.transpose(np.stack(
            [np.pad(x[3], ((0, 0), (0, max_target_length - x[5])), mode='constant') for x in batch]),
            axes=(0, 2, 1)
        )
        stop_token_target = np.stack(
            [np.pad(x[4],  (0, max_target_length - x[5]), mode='constant', constant_values=1.) for x in batch])

        return inputs, input_length, mel_target, linear_target, stop_token_target

    @staticmethod
    def load_one(item: pd.DataFrame):
        # input, input_length, mel_target, linear_target, stop_token_target
        inputs = np.asarray(text2sequence(item['pinyin']), dtype=np.int32)
        input_length = len(inputs)
        mel_target = np.log(np.maximum(np.load(item['mel_filepath']), 1e-5)) - hp.mel_log_center
        linear_target = np.log(np.maximum(np.load(item['spec_filepath']), 1e-5)) - hp.linear_log_center
        stop_token_target = np.asarray([0.] * mel_target.shape[-1])
        return inputs, input_length, mel_target, linear_target, stop_token_target, linear_target.shape[-1]


def main():
    mini_batch = MiniBatch(corpus_name='biaobei', batch_size=32)
    for i in range(10000//32):
        data = mini_batch.next_batch()
        print(i, ' batch:')
        for index, data_i in enumerate(data):
            print(index, ' data\'s shape is', data_i.shape)
        print('')

if __name__ == '__main__':
   main()
