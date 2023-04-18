#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 4/10/2023 4:56 PM
# @function  : the script is used to do something

import torch
import librosa
import utility as ut
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm

vocode_type = ['waveglow','parallel_waveglow_en','parallel_waveglow_jp','multi_band_melgan_en','multi_band_melgan_jp',
               'melgan_large','melgan','hifigan','full_band_melgan','tts']

vocoder_2_domain = [['waveglow','parallel_waveglow'],
                    ['multi_band_melgan','melgan_large','melgan','full_band_melgan'],
                    ['hifiGAN'],
                    ['voices_prompts_from_conformer_fastspeech2_pwg_ljspeech/generated']]

domain =['waveglow','melgan','hifiGAN','tts']

def genSpoof_list(dir_meta):
    label_dic = {}
    file_list = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        utt, base_data, alg = line.strip().split()
        file_list.append(utt)
        # TODO: one hot process
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

def get_dataset(cfg):
    print('Load Source Data')
    real_domain = get_real_domain_data()
    fake_domain_1, fake_domain_2, fake_domain_3, fake_domain_4 = get_fake_domain_data('/home/zhongjiafeng/Model/N801/key/wavefake-key/wavefake_key.txt')
    print('The real:fake_1:fake_2:fake_3:fake_4 = {}:{}:{}:{}:{}'.format(len(real_domain),len(fake_domain_1),len(fake_domain_2),
                                                                         len(fake_domain_3),len(fake_domain_4)))

    real_dataloader = DataLoader(real_domain,batch_size=cfg.batch_size, shuffle=True)
    fake_dataloader_1 = DataLoader(fake_domain_1,batch_size=cfg.batch_size, shuffle=True)
    fake_dataloader_2 = DataLoader(fake_domain_2, batch_size=cfg.batch_size, shuffle=True)
    fake_dataloader_3 = DataLoader(fake_domain_3, batch_size=cfg.batch_size, shuffle=True)
    fake_dataloader_4 = DataLoader(fake_domain_4, batch_size=cfg.batch_size, shuffle=True)
    return real_dataloader,fake_dataloader_1, fake_dataloader_2, fake_dataloader_3, fake_dataloader_4

def get_real_domain_data():
    ljspeech_base = '/home/zhongjiafeng/Model/N801/key/wavefake-key/ljspeech_key.txt'
    just_base = '/home/zhongjiafeng/Model/N801/key/wavefake-key/just_key.txt'
    real_path_list = []

    with open(ljspeech_base,'r') as f:
        lj_all_utt = f.readlines()

    with open(just_base,'r') as f:
        just_all_utt = f.readlines()

    all_utt = lj_all_utt + just_all_utt
    for meta in all_utt:
        real_path, label = meta.strip().split()
        real_path_list.append(real_path)

    real_dataset = AudioDataset(real_path_list, type='bonafide')
    return real_dataset
def get_fake_domain_data(base):
    fake_path_list = [[],[],[],[]]

    with open(base,'r') as f:
        all_utt = f.readlines()

    for utt in all_utt:
        utt_path, data_source, vocoder = utt.strip().split()
        for i,domain in enumerate(vocoder_2_domain):
            if vocoder in domain:
                fake_path_list[i].append(utt_path)

    fake_domain_dataset_1 = AudioDataset(fake_path_list[0],type='fake')
    fake_domain_dataset_2 = AudioDataset(fake_path_list[1], type='fake')
    fake_domain_dataset_3 = AudioDataset(fake_path_list[2], type='fake')
    fake_domain_dataset_4 = AudioDataset(fake_path_list[3], type='fake')

    return fake_domain_dataset_1, fake_domain_dataset_2, fake_domain_dataset_3,fake_domain_dataset_4

class AudioDataset(Dataset):
    def __init__(self, path_list, type='bonafide'):
        self.path_list = path_list
        self.label = type
        self.fixed_length = 64600 # 对应于4s

    def __getitem__(self, item):
        x, fs = librosa.load(self.path_list[item],sr=None)
        x_pad = ut.audio_pad(x,self.fixed_length)
        x = torch.Tensor(x_pad)
        label = 1 if self.label=='bonafide' else 0
        return x,label

    def __len__(self):
        return len(self.path_list)


if __name__ == '__main__':
    pass


