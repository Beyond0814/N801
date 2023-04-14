#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 2:43 PM
# @function  : the script is used to do something

import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset,DataLoader
import random
import os

def produce_evaluation_file(cm_score, file_list, protocol_dir, save_path):
    label_dic, meta = genSpoof_list(dir_meta=protocol_dir,return_meta=True)
    with open(save_path, 'a+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {} {} {}\n'.format(s, label_dic[utt], utt, meta[utt]))
    fh.close()
    print('Scores saved to : {}'.format(save_path))

def get_dataset(data_config, type=None):
    protocol = data_config.protocol[type]
    labels, file_train = genSpoof_list(dir_meta=protocol)
    train_set = Dataset_eval(
        config=data_config,
        file_name_list=file_train,
        label_list=labels,
    )
    return train_set

def get_FMFCC_dataloader(data_config, train_config):
    protocol = data_config.protocol[type]
    labels,file_train =genSpoof_list(dir_meta=protocol)
    print('no. of {} trials',len(file_train))
    train_set = Dataset_eval(
        config = data_config,
        file_name_list = file_train,
        label_list = labels,
    )

    train_loader = DataLoader(
        train_set,
        batch_size = train_config.batch_size,
        num_workers= train_config.num_workers,
        shuffle= train_config.shuffle,
        drop_last= train_config.drop_last
    )
    return train_loader


def genSpoof_list(dir_meta, return_meta=False):
    meta = {}
    label_dic = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        utt, label, alg = line.strip().split(',')

        file_list.append(utt)
        label_dic[utt] = 1 if label == '1' else 0
        meta[utt] = alg
    if return_meta:
        return label_dic, meta
    else:
        return label_dic, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        start = random.randint(0,x_len-max_len)
        return x[start:start+max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_eval(Dataset):
    '''
        ASVspoof19 train dataset.
    '''
    def __init__(self, eval_config, dataset_meta, file_name_list, label_list):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''
        self.file_name_list = file_name_list
        self.label_list = label_list
        self.base_dir = dataset_meta.base_dir
        self.cut = eval_config.cut_length

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        x, fs = librosa.load(self.base_dir + utt, sr=16000)
        x_pad = pad(x, self.cut)
        x = Tensor(x_pad)
        label = self.label_list[utt]

        return x, utt




