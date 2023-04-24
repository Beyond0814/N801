#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 4/18/2023 3:11 PM
# @function  : the script is used to do something
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import logging
from tqdm import tqdm
import evaluation.eval_metric as ev
from omegaconf import OmegaConf
import utility as ut
import numpy as np
# ===============================
import dataset_tool.fmfcc as data
import models.collection.aasist.wav2vec_aasist as Model
# ===============================
logger = logging.getLogger(__name__)
cfg = OmegaConf.load('./pseudo_config.yaml')


def run_pseudo_train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model.model(cfg,device)
    model = model.to(device)
    model.load_state_dict(torch.load(cfg.ckp_path, map_location=device))
    if cfg.num_gpus > 1:
        model = DataParallel(model).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    eval_loader = data.get_FMFCC_dataloader(cfg['FMFCC_A'],cfg.eval)

    for epoch in range(cfg.max_epoch):
        sub_train_list, score_path = get_top_confidence_sample(model, eval_loader, device, sample_num=cfg.sample_num)
        protocol_path = '/home/zhongjiafeng/Model/N801/key/FMFCC-A-keys/FMFCC-key.txt'
        eer, _ = ev.eval_base_protocol_and_score_file(score_path, protocol_path, 1, 1)
        logger.info('epoch [{}] - EER : {} '.format(epoch ,eer*100))

        train_loader = get_train_loader_from_confidence_sample(sub_train_list,cfg['FMFCC_A'],cfg.train)

        for iter in range(cfg.max_iter):
            model.train()
            train_num_total = 0.0
            train_loss = 0.0
            for batch_x, batch_y in tqdm(train_loader, desc='train', ncols=100):
                batch_size = batch_x.size(0)
                train_num_total += batch_size

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_output = model(batch_x)

                batch_loss = criterion(batch_output, batch_y)
                train_loss += (batch_loss.item() * batch_size)

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

            logger.info('epoch [{}] - iter [{}] : train loss {} \n'.format(epoch ,iter,train_loss))



def get_top_confidence_sample(model, eval_loader, device, sample_num):
    torch.cuda.empty_cache()
    model.eval()
    all_confidence = []
    all_utt = []
    all_scores = []
    with torch.no_grad():
        for batch_x, batch_y in tqdm(eval_loader, desc='eval', ncols=100):
            batch_x = batch_x.to(device)
            batch_out = model(batch_x)

            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
            confidence_out = -(batch_out.softmax(1) * batch_out.log_softmax(1)).sum(1)
            all_confidence.extend(confidence_out.data.cpu())
            all_utt.extend(batch_y)
            all_scores.extend(batch_score.tolist())
    all_confidence = np.array(all_confidence)
    all_utt = np.array(all_utt)
    indices = np.argsort(all_confidence)
    selected_indices = indices[:sample_num]
    selected_file_name = all_utt[selected_indices]

    score_path = os.path.join(os.getcwd(),'eval_confidence_scores.txt')
    ut.produce_scores_file(all_scores, all_utt, score_path)

    return selected_file_name, score_path

def get_train_loader_from_confidence_sample(file_list, data_config, train_config):
    protocol = data_config.protocol
    labels_dic, file_name_list = data.genSpoof_list(dir_meta=protocol)
    print('high-level Confidence dataset has {} trials.\n'.format(len(file_list)))
    eval_set = data.FMFCC_dataset(data_config, file_list, labels_dic, train_mode=True)

    train_loader = data.DataLoader(
        eval_set,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        shuffle=False,
    )
    return train_loader

if __name__ == '__main__':
    run_pseudo_train()
    print('------------------- Finish -------------------')