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
from utility_tool.utility import audio_pad

def produce_evaluation_file(cm_score, file_list, save_path):
    with open(save_path, 'a+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {:.12f}\n'.format(utt, s))
    fh.close()
    print('Scores saved to : {}'.format(save_path))

def get_dataloader(train_config, base_dir, protocol):
    assert type in ['train', 'dev'], 'parameter type is not train or dev.'
    label_dic, file_train = genSpoof_list(
        dir_meta=protocol,
        is_train=True,
        is_eval=False
    )
    train_set = ADD2023_train(train_config,base_dir,file_train,label_dic)

    train_loader = DataLoader(
        train_set,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
    )
    return train_loader

def get_eval_dataloader(data_config, eval_config):
    """
        用于获得测试集的dataloader
    :param data_config: 协议路径、数据路径等信息
    :param eval_config: dataloader等信息
    :return:
    """
    file_list = genSpoof_list(
        dir_meta=data_config.protocol,
        is_train=False,
        is_eval=True
    )
    eval_dataset = ADD2023_eval(
        eval_config=eval_config,
        data_config=data_config,
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



class ADD2023_train(Dataset):
    def __init__(self, train_config, base_dir, file_name_list, label_dic):
        self.file_name_list = file_name_list
        self.label_dic = label_dic
        self.base_dir = base_dir
        self.normalization = train_config.normalization

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        x, fs = librosa.load(self.base_dir + utt, sr=None)
        x_pad = audio_pad(x)
        x = Tensor(x_pad)
        if self.normalization:
            x = self.__normalize(x)
        label = self.label_dic[utt]
        return x, label

    def __normalize(self,wav):
        mean = wav.mean()
        std = wav.std()
        wav_normalize = (wav - mean) / std
        wav_normalize = torch.clamp(wav_normalize, min=-1, max=1)
        return wav_normalize

class ADD2023_eval(Dataset):
    def __init__(self, eval_config, data_config, file_name_list):
        self.file_name_list = file_name_list
        self.base_dir = data_config.base_dir
        self.fea_dir = data_config.feature_dir
        self.normalization = eval_config.normalization

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        x, fs = librosa.load(self.base_dir + utt, sr=None)
        x_pad = audio_pad(x)
        x = Tensor(x_pad)
        if self.normalization:
            x = self.__normalize(x)
        return x, utt

    def __normalize(self,wav):
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