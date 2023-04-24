#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 2:43 PM
# @function  : the script is used to do something

import numpy as np
import torch
from utility import audio_pad
from torch import Tensor
import librosa
from torch.utils.data import Dataset,DataLoader
import evaluation.eval_metric as ev
import pandas
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
    train_set = FMFCC_dataset(data_config,file_train,labels,)
    return train_set

def get_FMFCC_dataloader(data_config, train_config):
    protocol = data_config.protocol
    labels_dic,file_name_list =genSpoof_list(dir_meta=protocol)
    print('FMFCC dataset has {} trials.\n'.format(len(file_name_list)))
    eval_set = FMFCC_dataset(data_config, file_name_list,labels_dic)

    eval_loader = DataLoader(
        eval_set,
        batch_size = train_config.batch_size,
        num_workers= train_config.num_workers,
        shuffle= False,
    )
    return eval_loader


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

class FMFCC_dataset(Dataset):
    def __init__(self, data_config,file_name_list, label_dic, train_mode=False):
        '''self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)'''
        self.file_name_list = file_name_list
        self.label_dic = label_dic
        self.base_dir = data_config.base_dir
        self.cut = 64600
        self.train_mode = train_mode

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        utt = self.file_name_list[index]
        x, fs = librosa.load(self.base_dir + utt, sr=None)
        x_pad = audio_pad(x, self.cut)
        x = Tensor(x_pad)
        label = self.label_dic[utt]
        if self.train_mode:
            return x, label
        else:
            return x, utt

def analysis_score_file(file):
    meta = pandas.read_csv(file, sep=' ', header=None)
    target = pandas.DataFrame()
    #### change this to select which sample be evalueted ####
    alg_type = ['N01','A01','A02','A03','A04','A05','A06','A07','A08','A09','A10','A11','A12','A13']
    post_process = []
    for t in alg_type:
        post_process.append('{}_MP3'.format(t))
        post_process.append('{}_AAC'.format(t))
        post_process.append('{}_ND1'.format(t))
        post_process.append('{}_ND02'.format(t))

    for t in post_process:
        tmp = meta[meta[3] == t]
        target = pandas.concat([target,tmp], axis=0, ignore_index=True)
    #########################################################
    bona_cm = target[target[1] == 1][0].values
    spoof_cm = target[target[1] == 0][0].values
    eer_cm, threshold = ev.compute_eer(bona_cm, spoof_cm)

    print("whole type eer: {}".format(eer_cm * 100))
    print("whole type threshold: {}".format(threshold))
    return eer_cm * 100


