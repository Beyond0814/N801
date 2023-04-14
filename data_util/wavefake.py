#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 4/10/2023 4:56 PM
# @function  : the script is used to do something

import torch
import librosa
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

str_to_label = {

}

def genSpoof_list(dir_meta):
    label_dic = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        utt, base_data, alg = line.strip().split()
        file_list.append(utt)
        # TODO: one hot process
        label_dic[utt] = str_to_label[alg]

    return label_dic, file_list


def produce_label_file(base):
    subdir = os.listdir(base)
    label_file = open(base + 'wavefake_key.txt', 'a+')
    num = 0
    for dir in subdir:
        if len(dir) > 40:
            dir = os.path.join(dir,'generated')
        curdir = os.path.join(base,dir)

        if not os.path.isdir(curdir):
            continue
        else:
            print('Process the sub direction : {} '.format(curdir))
        base_data, alg = dir.strip().split('_',1)
        file_name_list = os.listdir(curdir)
        for utt in tqdm(file_name_list):
            utt_path =  os.path.join(curdir,utt)
            label_file.write('{} {} {}\n'.format(utt_path, base_data, alg))
            num += 1

    print('{} sample in this dataset.'.format(num))
    label_file.close()
    print('Finish.')

if __name__ == '__main__':
    pass