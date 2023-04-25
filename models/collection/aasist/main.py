import random
import os
from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from models.wavlm.WavLM import WavLM,WavLMConfig
from omegaconf import OmegaConf
from models.reference.resnet import ResNet
from fairseq import checkpoint_utils
from models.reference.lcnn import LCNN9

all_model = ['Wavlm-ResNet','wav2vec2.0-FC']

class wav2vec(nn.Module):
    def __init__(self, model_config, device):
        super(wav2vec, self).__init__()

        cp_path = model_config.ssl_path # Change the pre-trained XLSR model path.
        model, cfg, task = checkpoint_utils.load_model_ensemble_and_task([cp_path],suffix="")
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        self.freeze = model_config.ssl_freeze
        self.ssl_name = model_config.ssl_name

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            print('freeze the parameters of ssl model.')

        nb_params = sum([param.view(-1).size()[0] for param in self.model.parameters()])
        print('{} model have nb_params: {}'.format(self.ssl_name, nb_params))
        return

    def extract_feat(self, input_data):

        # put the model to GPU if it not there
        # if next(self.model.parameters()).device != input_data.device \
        #    or next(self.model.parameters()).dtype != input_data.dtype:
        self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

        if True:
            # input should be in shape (batch, length)
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data

            # [batch, length, dim]
            emb = self.model(input_tmp, mask=False, features_only=True)['x']  # 模型输出是个字典
        return emb

class Wavlm(nn.Module):
    def __init__(self, model_config, device):
        super(Wavlm, self).__init__()
        self.freeze = model_config.ssl_freeze
        self.device = device
        self.ssl_name = model_config.ssl_name
        self.out_dim = None
        self.cfg = None


        cp_path = "/pubdata/zhongjiafeng/self_supervise_model/WaveLM/WavLM-Base.pt"
        checkpoint = torch.load(cp_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg).to(device)
        model.load_state_dict(checkpoint['model'])
        self.model = model
        self.cfg = cfg
        self.out_dim = 1024

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            print('freeze the parameters of ssl model.')

        nb_params = sum([param.view(-1).size()[0] for param in self.model.parameters()])
        print('{} model have nb_params: {}'.format(self.ssl_name, nb_params))

        return

    def extract_feat(self, input_data):
        self.model.to(input_data.device, dtype=input_data.dtype)
        self.model.train()

        if self.cfg.normalize:
            input_data = torch.nn.functional.layer_norm(input_data, input_data.shape)

        rep = self.model.extract_features(input_data)[0]
        return rep

class wav2vec_resnet(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.ssl_model = wav2vec(model_config,self.device)
        self.resnet = ResNet(num_nodes=4, enc_dim=256)

    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x) # -> [batch, time_frame, 1024]
        x = x_ssl_feat.unsqueeze(1)
        x = x.transpose(2,3) #  -> [batch,1,1024,time_frame]
        _ ,x = self.resnet(x)
        return x

class wav2vec_lcnn(nn.Module):
    def __init__(self, model_config, device):
        super().__init__()
        self.device = device
        self.ssl_model = wav2vec(model_config, self.device)
        self.lcnn = LCNN9(out_dim=2)

    def forward(self, x):
        x_ssl_feat = self.ssl_model.extract_feat(x) # -> [batch, time_frame, 768]
        x = x_ssl_feat.unsqueeze(1)
        x = self.lcnn(x)
        return x

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = OmegaConf.load('/home/zhongjiafeng/Model/N801/config/ADD2023_config.yaml')
    Model = wav2vec_lcnn(model_config=cfg.model_configuration,device=device).to(device)
    print(Model)
    input = torch.randn(2,64600).to(device)
    output = Model(input)
    print(output.size())

