#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 10:54 AM
# @function  : the script is used to do something

import librosa
import audioread
import random
import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset,DataLoader
from pydub import AudioSegment

def gen_Spoof_Bonafide_list(dir_meta, dataset):
    d_meta = {}
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    if dataset == 'ASVspoof19':
        spoof_list = []
        bonafide_list = []
        for line in l_meta:
            _, utt, _, _, label = line.strip().split()
            if label == 'bonafide':
                bonafide_list.append(utt)
            else:
                spoof_list.append(utt)
            d_meta[utt] = 1 if label == 'bonafide' else 0
        return d_meta, spoof_list, bonafide_list
    else:
        print('error.')


class Compressor():
    def __init__(self, dataset, base_dir, protocol, target_dir):
        self.dataset = dataset
        self.protocol = protocol
        self.format = 'flac'
        self.base_dir = base_dir
        self.target_dir = target_dir
        self.toy_base = os.path.join(self.target_dir,'toy_large/')
        self.sam_protocol_path = os.path.join(self.toy_base, 'sam_from_{}_label.txt'.format(self.dataset))
        self.toy_file_list = None
        self.label_dic = None

    def sample_from_dataset(self, num):
        label_dic, spoof_list, bonafide_list = gen_Spoof_Bonafide_list(self.protocol,self.dataset)
        s_num = int(num * 0.9)
        b_num = int(num * 0.1)
        spoof_sampler = random.sample(spoof_list,s_num)
        bonafide_sampler = random.sample(bonafide_list,b_num)
        sampler = np.concatenate([bonafide_sampler, spoof_sampler], axis=0)
        self.toy_file_list = sampler
        self.label_dic = label_dic

        if not os.path.exists(self.toy_base + 'wav'):
            os.makedirs(self.toy_base + 'wav')
        for s in tqdm(sampler,desc='Converting :', ncols=100):
            src = AudioSegment.from_file(self.base_dir + '{}.{}'.format(s,self.format),self.format)
            src.export(self.toy_base + 'wav/{}.wav'.format(s), "wav")

        with open(self.sam_protocol_path,'a+') as f:
            f.write('sample from {}, {} number of samples.\n'.format(self.dataset, num))
            for utt in sampler:
                f.write('{} {}\n'.format(utt, label_dic[utt]))

            print("protocol file create in {}".format(self.sam_protocol_path))

    def post_precess(self, command_line: list, name):
        # TODO: process each one wave file and save them
        if self.toy_file_list == None:
            self.label_dic, self.toy_file_list = gen_toy_list(self.sam_protocol_path)

        save_dir = self.target_dir + 'toy_{}/'.format(name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for utt in tqdm(self.toy_file_list):
            src = AudioSegment.from_file(self.toy_base + 'wav/{}.wav'.format(utt, self.format), 'wav')
            src.export(save_dir+utt+'.mp3', format='mp3',parameters=command_line)

        print('process finish.')

def gen_toy_list(sam_protocol_path):
    label_dic = {}
    toy_file_list = []
    with open(sam_protocol_path, 'r') as f:
        l_meta = f.readlines()

    print(l_meta[0])
    for line in l_meta[1:]:
        utt, label = line.strip().split()
        toy_file_list.append(utt)
        label_dic[utt] = 1 if label == '1' else 0
    return label_dic, toy_file_list

def check_same(a_path, b_path):
    a, asr = librosa.load(a_path,sr=None)
    b, bsr = librosa.load(b_path,sr=None)
    if len(a)==len(b) and (a==b).all():
        print('same.')
    else:
        print('different.')
    return

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class toy_dataset(Dataset):
    def __init__(self, config, file_name_list, base_dir):
        self.file_name_list = file_name_list
        self.base_dir = base_dir
        self.cut = config.cut_length

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, item):
        utt = self.file_name_list[item]
        x, _ = librosa.load(self.base_dir+ utt + '.ogg', sr=None)
        x_pad = pad(x, self.cut)
        x_inp = torch.Tensor(x_pad)
        return x_inp, utt

def produce_evaluation_file(cm_score, file_list, protocol_dir, save_path):
    label_dic, toy_file_list = gen_toy_list(protocol_dir)
    with open(save_path, 'a+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {} {}\n'.format(s, label_dic[utt], utt))
    fh.close()
    print('Scores saved to : {}'.format(save_path))



if __name__ == '__main__':
    dataset = 'ASVspoof19'
    base_dir = "/pubdata/zhongjiafeng/ASVspoof_LA/ASVspoof2019_LA_eval/flac/"
    protocol = '/home/zhongjiafeng/Model/N801/database/19LA-keys/ASVspoof2019.LA.cm.eval.trl.txt'
    target_dir =  "/pubdata/zhongjiafeng/Compressdataset/"
    c = Compressor(dataset,base_dir, protocol, target_dir)
    # c.sample_from_dataset(10000)
    mp3 = ['128k','64k','32k','16k','8k']
    ogg = ['16k','32k','64k','96k']
    aac = ['96k','64k','32k','16k','8k','4k']
    param = ["-c:a","libmp3lame", "-b:a", "32k"]
    fmt = '32k'
    c.post_precess(param, 'large_mp3_{}'.format(fmt))
    print('OK')