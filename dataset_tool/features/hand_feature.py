#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 2022/10/21 9:56
# @function  : the script is used to configure hyperparameter which is used during training model. Parameters
#              need to configure are as follows:
#          由于每种特征的长度都不一样，所以改变特征时，应对模型输入输出维度进行微调

import soundfile as sf
import librosa
import torch
from librosa.core.spectrum import _spectrogram, power_to_db
from librosa.feature.spectral import melspectrogram
from spafe.features import lfcc
from data_util.features.util import gtcc, LFB
from omegaconf import OmegaConf
from tqdm import tqdm
import os
import numpy as np

def hand_feature_extraction(sign, fs, feature_name, feature_config):
    feature = None
    hyper = feature_config[feature_name]

    if feature_name == 'MFCC':
        feature = librosa.feature.mfcc(
            y=sign, sr=fs, n_mfcc=hyper['num_ceps'], lifter=hyper['lifter'], hop_length=hyper['hop_length'],
            win_length=hyper['win_length'], window=hyper['window'], center=hyper['center'], pad_mode=hyper['pad_mode'],
            n_fft=hyper['n_fft'], n_mels=hyper['nfilts'],
        )
        feature = feature.T

    elif feature_name == 'MSTFT':
        feature = melspectrogram(
            y=sign, sr=fs, hop_length=hyper['hop_length'], power=hyper['power'],
            win_length=hyper['win_length'], window=hyper['window'], center=hyper['center'], pad_mode=hyper['pad_mode'],
            n_fft=hyper['n_fft'], n_mels=hyper['nfilts'],
        )
        feature = feature.T
        delta = lfcc.delta(feature)
        delta_delta = lfcc.delta(delta)
        feature = np.concatenate((feature, delta, delta_delta), axis=1)

    elif feature_name == 'LFCC':
        feature = lfcc.lfcc(
            sign, fs=fs, num_ceps=hyper['num_ceps'],  pre_emph=hyper['pre_emph'],nfilts=hyper['nfilts'], nfft=hyper['nfft'],
            low_freq=hyper['low_freq'], scale=hyper['scale'], dct_type=2,
            use_energy=hyper['use_energy']
        )

    elif feature_name == 'GTCC':
        feature = gtcc(
            sign, fs=fs, pre_emph=hyper['pre_emph'], win_len=hyper['win_len'], win_hop=hyper['win_hop'],
            win_type=hyper['win_type'], nfilts=hyper['nfilts'], nfft=hyper['nfft'], low_freq=hyper['low_freq'],
            high_freq=hyper['high_freq'], scale=hyper['scale'], dct_type=2, order=4, num_ceps=hyper['num_ceps'],
        )

    elif feature_name == 'LFBs':
        feature = LFB(
            sign, fs=fs, pre_emph=hyper['pre_emph'], win_len=hyper['win_len'],
            win_hop=hyper['win_hop'], win_type=hyper['win_type'], nfilts=hyper['nfilts'], nfft=hyper['nfft'],
            low_freq=hyper['low_freq'], high_freq=hyper['high_freq'], scale=hyper['scale'], mode='LFBs',
        )

    elif feature_name == 'LPS':
        spectrum, _ = _spectrogram(
            y=sign, n_fft=hyper['n_fft'], hop_length=hyper['hop_length'], win_length=hyper['win_length'],
            window=hyper['window'], center=hyper['centre'], pad_mode=hyper['pad_mode'], power=hyper['power'],
        )
        feature = power_to_db(spectrum)

    elif feature_name == 'LLFB':
        feature = LFB(
            sign, fs=fs, pre_emph=hyper['pre_emph'], win_len=hyper['win_len'],
            win_hop=hyper['win_hop'], win_type=hyper['win_type'], nfilts=hyper['nfilts'], nfft=hyper['nfft'],
            low_freq=hyper['low_freq'], high_freq=hyper['high_freq'], scale=hyper['scale'], mode='LLFB',
        )
    # 返回特征统一格式为ndarray 且帧沿第一个轴排列,特征沿第二个轴排列，即每行为一个帧的特征
    return feature

def produce_feature_cache(yaml_path,feature_name,save_dir,type):
    config = OmegaConf.load(yaml_path)
    if type == 'eval':
        base_dir = config.database.database_path + 'wav/'
        protocol = '/home/zhongjiafeng/Model/N801/database/ADD2023-key/Track1_2_eval_key.txt'
    else:
        base_dir = config.database.database_path + '{}/wav/'.format(type)
        protocol = config.database.protocol_path + '{}/wavefake_key.txt'.format(type)
    current_path = os.path.dirname(__file__)
    config = OmegaConf.load('{}/hyper.yaml'.format(current_path))

    file_name_list = []
    with open(protocol, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        if type in ['dev','train']:
            utt, label = line.strip().split()
        else:
            utt = line.strip()

        file_name_list.append(utt)

    for name in tqdm(file_name_list):
        sign, fs = sf.read(base_dir + name)
        f = hand_feature_extraction(sign, fs, feature_name, config)
        f = torch.from_numpy(f)
        torch.save(f, save_dir + name)

    print('finish.')

def run():
    set = ['train', 'dev', 'eval']
    mode = 2
    yaml_path = '/home/zhongjiafeng/Model/N801/config/ADD2023_config.yaml'
    feature_name = 'LFCC'
    save_dir = '/pubdata/zhongjiafeng/ADD_cache/{}/'.format(set[mode])
    produce_feature_cache(yaml_path, feature_name, save_dir, set[mode])

if __name__ == '__main__':
    sign, fs = sf.read('./example.flac')
    current_path = os.path.dirname(__file__)
    config = OmegaConf.load('{}/hyper.yaml'.format(current_path))
    raw_lengthN = int(4 * fs)  # 固定后语音信号的点数
    if len(sign) < raw_lengthN:  # 长度不够则将数据重复，然后取为固定长度
        sign = np.tile(sign, int((raw_lengthN) // len(sign)) + 1)
    sign = sign[0:raw_lengthN]
    sign = np.reshape(sign, (sign.shape[0], 1))

    f = hand_feature_extraction(sign, fs, 'LFCC', config)
    print(f)

