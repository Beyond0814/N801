#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/21/2023 10:48 AM
# @function  : the script is used to do something

import numpy as np
import torch
from torch import Tensor
import librosa
from torch.utils.data import Dataset, DataLoader , DistributedSampler

def produce_evaluation_file(cm_score, file_list, save_path):
    with open(save_path, 'a+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {:.12f}\n'.format(utt, s))
    fh.close()
    print('Scores saved to : {}'.format(save_path))

def get_dataloader(data_config, train_config, type=None):
    assert type in ['train', 'dev'], 'parameter type is not train or dev.'
    protocol = data_config.protocol[type]
    labels, file_train = genSpoof_list(
        dir_meta=protocol,
        is_train=True,
        is_eval=False
    )
    train_set = Dataset_train(
        config=data_config,
        file_name_list=file_train,
        label_list=labels,
        type=type
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=train_config.shuffle,
        drop_last=train_config.drop_last,
        pin_memory=True
    )
    return train_loader

def get_eval_dataloader(data_config, eval_config):
    file_list = genSpoof_list(
        dir_meta=data_config.protocol,
        is_train=False,
        is_eval=True
    )
    eval_dataset = Dataset_eval(
        eval_config=eval_config,
        dataset_meta=data_config,
        file_name_list=file_list,
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        shuffle=False,
    )
    return eval_loader

def genSpoof_list(dir_meta, is_train=False, is_eval=False):
    d_meta = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if (is_train):
        for line in l_meta:
            utt, label = line.strip().split()

            file_list.append(utt)
            d_meta[utt] = 1 if label == 'genuine' else 0
        return d_meta, file_list

    elif (is_eval):
        for line in l_meta:
            utt = line.strip()

            file_list.append(utt)
        return file_list


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
        assert type in ['train', 'dev']
        self.file_name_list = file_name_list
        self.label_list = label_list
        self.base_dir = config.database_path + '{}/wav/'.format(type)
        self.fea_dir = config.feature_path + '{}/'.format(type)
        self.cut = config.cut_length
        self.DA = config.enable_data_augment_in_train
        self.hand_feature = config.hand_feature
        self.normalization = config.normalization

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        if self.hand_feature:
            x = torch.load(self.fea_dir + utt)
            r = (self.cut//x.shape[0]) + 1
            x = x.repeat(r,1)
            x = x[:self.cut,:].float()
        else:
            x, fs = librosa.load(self.base_dir + utt, sr=16000)
            x_pad = pad(x, self.cut)
            x = Tensor(x_pad)
            if self.normalization:
                x = self.normalize(x)
        label = self.label_list[utt]
        return x, label

    def normalize(self,wav):
        mean = wav.mean()
        std = wav.std()
        wav_normalize = (wav - mean) / std
        wav_normalize = torch.clamp(wav_normalize, min=-1, max=1)
        return wav_normalize

class Dataset_eval(Dataset):
    def __init__(self, eval_config, dataset_meta, file_name_list):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''
        self.file_name_list = file_name_list
        self.base_dir = dataset_meta.base_dir
        self.cut = eval_config.cut_length
        self.fea_dir = dataset_meta.feature_dir
        self.hand_feature = eval_config.hand_feature
        self.normalization = eval_config.normalization

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        if self.hand_feature:
            print('using hand feature as input tensor.')
            x = torch.load(self.fea_dir + utt)
            r = (self.cut//x.shape[0]) + 1
            x = x.repeat(r,1)
            x = x[:self.cut,:].float()
        else:
            x, fs = librosa.load(self.base_dir + utt, sr=16000)
            x_pad = pad(x, self.cut)
            x = Tensor(x_pad)
            if self.normalization:
                x = self.normalize(x)
        return x, utt

    def normalize(self,wav):
        mean = wav.mean()
        std = wav.std()
        wav_normalize = (wav - mean) / std
        wav_normalize = torch.clamp(wav_normalize, min=-1, max=1)
        return wav_normalize

def extract_label_from_mos(type, fake_num):

    mos_file = open('/home/zhongjiafeng/Model/N801/database/ADD2023-key/mos_label/{}_fake_mos.txt'.format(type),'r')
    label_file = open('/pubdata/zhongjiafeng/ADD2023/Track1/track1.2/{}/label.txt'.format(type),'r')

    mos_list = mos_file.readlines()
    label_list = label_file.readlines()

    mos_file.close()
    label_file.close()

    output_file = open('/home/zhongjiafeng/Model/N801/database/ADD2023-key/large/{}_top_mos_label.txt'.format(type),'w')
    count = 0
    for meta in mos_list:
        mos, utt = meta.strip().split()
        output_file.write('{} {}\n'.format(utt, 'fake'))
        count += 1
        if count == fake_num:
            break

    for meta in label_list:
        utt, label = meta.strip().split()
        if label == 'genuine':
            output_file.write('{} {}\n'.format(utt, label))

    print('finish.')

if __name__ == '__main__':
    extract_label_from_mos('dev',10000)