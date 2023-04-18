#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/6/2023 10:45 PM
# @function  : the script is used to do something
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import librosa
from models import Model,Model_with_FC

def genSpoof_list_meta(dir_meta):
    d_meta = []
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        data= line.strip().split()
        d_meta.append(data)

    d_meta = np.array(d_meta)
    return d_meta

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len)+1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class Dataset_sample(Dataset):
    def __init__(self, meta,sample_num, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
           '''
        self.meta = meta
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.labels = meta[:,5]
        self.IDs = meta[:,1]
        spoof = np.argwhere(self.labels == 'spoof')
        bonafide = np.argwhere(self.labels == 'bonafide')
        spoof_seed = np.random.randint(low=0, high=int(len(spoof)), size=int(sample_num/2))
        bonafide_seed = np.random.randint(low=0, high=int(len(bonafide)), size=int(sample_num/2))

        spoof = spoof[spoof_seed]
        bonafide = bonafide[bonafide_seed]
        self.sample_position =np.concatenate([spoof, bonafide], axis=0)

    def __len__(self):
        return len(self.sample_position)

    def __getitem__(self, index):
        t = int(self.sample_position[index])
        label = self.labels[t]
        X, fs = librosa.load(self.base_dir + 'flac/' + self.IDs[t] + '.flac', sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, label

def produce_evaluation_file_tsne(dataset, model, device):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    model.eval()
    embed_list = []
    label_list = []
    for batch_x, label in data_loader:
        batch_x = batch_x.to(device)
        batch_out, batch_embed = model(batch_x)
        batch_embed = batch_embed.data.cpu().numpy()
        # add outputs
        label_list.extend(label)
        embed_list.extend(batch_embed)

    embed_list = np.array(embed_list)
    label_list = np.array(label_list)
    return embed_list, label_list

def tSNE_analyse(data, label, path):
    result = TSNE(n_components=2,random_state=1).fit_transform(data)
    x_min, x_max = np.min(result,0), np.max(result,0)
    result = (result - x_min)/(x_max-x_min)
    for i in range(result.shape[0]):
        if label[i]=='spoof':
            s = plt.scatter(result[i,0], result[i,1],c='g',label='spoof')
        else:
            b = plt.scatter(result[i, 0], result[i, 1],c='b',label='bonafide')
    plt.xticks([])
    plt.yticks([])
    plt.legend(handles=[s,b],labels=['spoof','bonafide'])
    plt.savefig(path)


if __name__ == '__main__':
    meta_dir = '/pubdata/zhongjiafeng/ASVspoofLA/ASVspoof2021_LA_eval/keys/LA/CM/trial_metadata.txt'
    base_dir = '/pubdata/zhongjiafeng/ASVspoofLA/ASVspoof2021_LA_eval/'
    model_path = '/home/zhongjiafeng/Model/SSL/models/model_LA_weighted_CCE_10_8_1e-06_Model_FC_version/best_model_Model_FC_version.pth'
    sample_numb = 2000
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: {}'.format(device))
    meta = genSpoof_list_meta(meta_dir)
    print('no. of eval trials', len(meta))
    eval_set = Dataset_sample(meta=meta, sample_num=sample_numb, base_dir=base_dir)

    model = Model_with_FC(None, device)  # 模型
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model = model.to(device)
    print('nb_params:', nb_params)

    model.load_state_dict(torch.load(model_path, map_location=device))
    print('Model loaded : {}'.format(model_path))

    data, label = produce_evaluation_file_tsne(eval_set, model, device)
    tSNE_analyse(data, label, path='tsne_figure.png')
    print('tsne OK.')