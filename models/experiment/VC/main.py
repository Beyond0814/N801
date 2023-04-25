import sys
import os
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel

import utility as ut
import evaluation as ev
import numpy as np
import dataset_tool as dt


import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm

################   custom    ################
import dataset_tool.asvspoof as Data
import models as Model
################   custom    ################

log = logging.getLogger(__name__)


def train(rank, cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_configuration
    train_config = cfg.training
    data_config = cfg.database

    ################   custom    ################
    data_config.protocol['train'] = data_config.protocol_path + 'ASVspoof2019.LA.cm.train.trn.txt'
    data_config.protocol['dev'] = data_config.protocol_path + 'ASVspoof2019.LA.cm.dev.trl.txt'
    ################   custom    ################

    train_loader = Data.get_dataloader(data_config, train_config, type='train')
    dev_loader = Data.get_dataloader(data_config, train_config, type='dev')

    ################   custom    ################
    model = Model.model(model_config, device)
    ################   custom    ################
    model = model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])

    if model_config.load_checkpoint:
        model.load_state_dict(torch.load(model_config.model_path,map_location=device))
        log.info('Model loaded : {}'.format(model_config.model_path))

    if cfg.training.num_gpus > 1:
        model = DataParallel(model).to(device)

    if rank == 0:
        log.info('nb_params: {}'.format(nb_params))
        log.info(model)

    writer = SummaryWriter(os.getcwd())

    ################   custom    ################
    opt_config = train_config.optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt_config.base_lr, weight_decay=opt_config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lr_lambda=lambda step: ut.cosine_annealing(step,
                                                  train_config.max_num_epoch * len(train_loader),
                                                  opt_config))
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    ################   custom    ################

    tqdm_dis = {}
    for epoch in range(train_config.max_num_epoch):
        model.train()
        train_num_total = 0.0
        dev_num_total = 0.0
        train_loss = 0.0
        dev_loss = 0.0
        best_dev_loss = 99
        best_eer = 99
        tqdm_dis['lr']=optimizer.state_dict()['param_groups'][0]['lr']

        for batch_x, batch_y in tqdm(train_loader, desc='train', ncols=100 ,postfix=tqdm_dis):
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
            scheduler.step()

        train_loss = train_loss / train_num_total
        model.eval()
        cm_score = []
        label = []
        with torch.no_grad():
            for batch_x, batch_y in tqdm(dev_loader, desc='dev', ncols=100):
                batch_size = batch_x.size(0)
                dev_num_total += batch_size

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_output = model(batch_x)
                batch_score = (batch_output[:,1]).data.cpu().numpy().ravel()

                batch_loss = criterion(batch_output, batch_y)
                dev_loss += (batch_loss.item() * batch_size)

                cm_score.extend(batch_score.tolist())
                label.extend(batch_y.data.cpu().numpy().ravel().tolist())



            cm_score = np.array(cm_score).ravel()
            label = np.array(label).ravel()
            dev_loss = dev_loss / dev_num_total
            EER, _ = ev.calculate_EER_only(cm_score, label)
            log.info('epoch [{}] , train loss: {} , dev loss: {} , dev EER: {}'.format(epoch, train_loss, dev_loss, EER))
            writer.add_scalar('val_loss', dev_loss, epoch)
            writer.add_scalar('loss', train_loss, epoch)
            writer.add_scalar('lr', tqdm_dis['lr'], epoch)

            if EER < best_eer:
                best_eer = EER
                path = os.path.join(os.getcwd(), '{}_epoch_{}_eer_model.pth'.format(epoch, EER))
                torch.save(model.module.state_dict(), path)
                log.info('Flash the EER , save to {}'.format(path))

    # final save
    path = os.path.join(os.getcwd(), 'Final_model.pth')
    torch.save(model.module.state_dict(), path)


#################################-----main function-----###############################
@hydra.main(config_path='./config', config_name='me_config')
def main(cfg: DictConfig):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_VISIBLE
    available_gpu_num = torch.cuda.device_count()
    log.info('GPU number: {}'.format(available_gpu_num))

    ut.set_random_seed(cfg.random_seed)
    train(0, cfg)


if __name__ == '__main__':
    main()
    log.info("Congraduation! The Prgram runing finish.^_^!!!")