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
from .rawboost import process_Rawboost_feature

def get_dataset(data_config, type=None):
    assert type in ['train','dev'], 'parameter type is not train or dev.'
    protocol = data_config.protocol[type]
    labels,file_train =genSpoof_list(
        dir_meta=protocol,
        is_train=True,
        is_eval=False
                                    )

    train_set = Dataset_train(
        config = data_config,
        file_name_list = file_train,
        label_list = labels,
        type=type
    )
    return train_set

def get_dataloader(data_config, train_config, type=None):

    assert type in ['train','dev'], 'parameter type is not train or dev.'
    protocol = data_config.protocol[type]
    labels,file_train =genSpoof_list(
        dir_meta=protocol,
        is_train=True,
        is_eval=False
                                    )
    print('no. of {} trials'.format(type), len(file_train))
    train_set = Dataset_train(
        config = data_config,
        file_name_list = file_train,
        label_list = labels,
        type=type
    )

    train_loader = DataLoader(
        train_set,
        batch_size = train_config.batch_size,
        num_workers= train_config.num_workers,
        shuffle= train_config.shuffle,
        drop_last= train_config.drop_last
    )

    return train_loader


def genSpoof_list(dir_meta, is_train=False, is_eval=False, config_phase = None):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            _, utt, _, _, label = line.strip().split()

            file_list.append(utt)
            d_meta[utt] = 1 if label == 'bonafide' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, _, label, _, phase = line.strip().split()
            if config_phase != phase and config_phase!='all':
                continue
            file_list.append(key)
            d_meta[key] = 1 if label == 'bonafide' else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


class Dataset_train(Dataset):
    '''
        ASVspoof19 train dataset.
    '''
    def __init__(self, config, file_name_list, label_list, type=None):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''
        assert type in ['train','dev']
        self.file_name_list = file_name_list
        self.label_list = label_list
        self.base_dir = config.database_path + 'ASVspoof2019_LA_{}/'.format(type)
        self.cut = config.cut_length
        self.DA = config.data_augment

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        x, fs = librosa.load(self.base_dir + 'flac/' + utt + '.flac', sr=16000)
        if self.DA:
            x = process_Rawboost_feature(x,fs)
        x_pad = pad(x, self.cut)
        x = Tensor(x_pad)
        label = self.label_list[utt]

        return x, label

class Dataset_eval19LA(Dataset):
    def __init__(self, config, file_name_list, label_dic, base_dir):
        self.file_name_list = file_name_list
        self.base_dir = base_dir
        self.cut = config.cut_length
        self.label_dic = label_dic


    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + utt + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        label = self.label_dic[utt]
        return x_inp, utt

class Dataset_eval21LA(Dataset):
    def __init__(self, config, file_name_list, label_dic, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''
        self.file_name_list = file_name_list
        self.base_dir = base_dir
        self.cut = config.cut_length
        self.label_dic,_ = genSpoof_list(dir_meta = '/home/zhongjiafeng/Model/N801/database/21LA-keys/keys/CM/trial_metadata.txt',
                         is_train=False, is_eval=False)


    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt_id = self.file_name_list[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + utt_id + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        label = self.label_dic[utt_id]
        return x_inp, utt_id


class Dataset_eval21DF(Dataset):
    def __init__(self, config, file_name_list, label_dic, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''
        self.file_name_list = file_name_list
        self.base_dir = base_dir
        self.cut = config.cut_length
        self.label_dic, _ = genSpoof_list(
            dir_meta='/home/zhongjiafeng/Model/N801/database/21LA-keys/keys/CM/trial_metadata.txt',
            is_train=False, is_eval=False)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt_id = self.file_name_list[index]
        X, fs = librosa.load(self.base_dir + 'flac/' + utt_id + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        label = self.label_dic[utt_id]
        return x_inp, utt_id



