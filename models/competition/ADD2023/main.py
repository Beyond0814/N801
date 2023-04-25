import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'

import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel


import utility_tool.utility as ut
from utility_tool.loggers import log
import evaluation as ev
import numpy as np
import dataset_tool as dt



from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
################   custom import   ################
import dataset_tool.add as Data
import models.collection.aasist.wav2vec_aasist as Model

cfg = OmegaConf.load('./config.yaml')
def train(rank, cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_configuration
    train_config = cfg.training
    data_config = cfg.database

    data_config.protocol['train'] = data_config.protocol_path + 'train_top_mos_label.txt'
    data_config.protocol['dev'] = '/pubdata/zhongjiafeng/ADD2023/Track1/track1.2/dev/label.txt'

    train_loader = Data.get_dataloader(data_config, train_config, type='train')
    dev_loader = Data.get_dataloader(data_config, train_config, type='dev')

    model = Model.wav2vec_lcnn(model_config,device)
    model = model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model_config.model_params = nb_params

    if model_config.load_checkpoint:
        model.load_state_dict(torch.load(model_config.model_path,map_location=device))
        log.info('Model loaded : {}'.format(model_config.model_path))

    if cfg.training.num_gpus > 1:
        model = DataParallel(model).to(device)

    if rank == 0:
        log.info('nb_params: {}'.format(nb_params))
        log.info('=========================================================================')
        log.info(model)
        log.info('=========================================================================')

    writer = SummaryWriter(os.getcwd())

    opt_config = train_config.optimizer

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=opt_config.base_lr, weight_decay=opt_config.weight_decay)
    if train_config.optimizer.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda step:
        ut.cosine_annealing(step,train_config.max_num_epoch * len(train_loader),opt_config))

    elif train_config.optimizer.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=0.5,total_iters=200)

    else :
        print('scheduler error .')

    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    tqdm_dis = {}
    train_num_total = 0.0
    dev_num_total = 0.0
    train_loss = 0.0
    dev_loss = 0.0
    best_dev_loss = 99
    best_eer = 99
    for epoch in range(train_config.max_num_epoch):
        model.train()
        tqdm_dis['lr']=optimizer.state_dict()['param_groups'][0]['lr']
        ## Training
        for batch_x, batch_y in tqdm(train_loader, desc='[train]', ncols=150 ,postfix=tqdm_dis):
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
        log.info('epoch [{}] : train loss: {}'.format(epoch, train_loss))
        writer.add_scalar('loss', train_loss, epoch)
        writer.add_scalar('lr', tqdm_dis['lr'], epoch)

        if epoch % train_config.validation_interval == 0 :
            model.eval()
            cm_score = []
            label = []
            # development
            with torch.no_grad():
                for batch_x, batch_y in tqdm(dev_loader, desc='[ dev ]', ncols=150):
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
                log.info('epoch [{}] : dev loss: {} , dev EER: {}'.format(epoch, dev_loss, EER))
                writer.add_scalar('val_loss', dev_loss, epoch)

                # eval
                if dev_loss <= best_dev_loss:
                    best_dev_loss = dev_loss
                    log.info('epoch [{}] :Flash the best dev-loss : {}, save model.'.format(epoch,best_dev_loss))
                    path = os.path.join(os.getcwd(), '{}_best_dev_loss_model.pt'.format(model_config.model_name))
                    torch.save(model.module.state_dict(), path)
    # final save
    path = os.path.join(os.getcwd(), 'Final_model.pt')
    torch.save(model.module.state_dict(), path)

def evaluate(cfg):
    eval_config = cfg.evaluate
    eval_dataset_config = cfg.ADD2023_2
    model_config = cfg.model_configuration

    save_path = os.path.join(os.getcwd(), '{}_score_file.txt'.format(cfg.model_configuration.model_name))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Model.model(model_config,device)
    model = model.to(device)

    if cfg.num_gpus > 1:
        model = DataParallel(model).to(device)
    if model_config.load_checkpoint:
        model.module.load_state_dict(torch.load(model_config.model_path, map_location=device))
        log.info('Model loaded : {}'.format(model_config.model_path))

    eval_loader = Data.get_eval_dataloader(eval_dataset_config,eval_config)
    cm_score = []
    utt_list = []
    model.eval()
    with torch.no_grad():
        for batch_x, batch_utt in tqdm(eval_loader, ncols=150):
            batch_x = batch_x.to(device)
            batch_output = model(batch_x)
            batch_score = (batch_output[:, 1]).data.cpu().numpy().ravel()

            cm_score.extend(batch_score.tolist())
            utt_list.extend(batch_utt)

        cm_score = np.array(cm_score).ravel()
        utt_list = np.array(utt_list)

        ut.produce_scores_file(cm_score,utt_list,save_path)


#################################-----main function-----###############################
def main(cfg: DictConfig):
    available_gpu_num = torch.cuda.device_count()
    assert available_gpu_num == cfg.num_gpus,'cfg.num_gpus not equal with available_gpu_num.'
    log.info('GPU number: {}'.format(available_gpu_num))

    assert cfg.mode in ['train','eval'], 'invalid mode, please check config file.'
    if cfg.mode == 'train':
        # Read configuration file
        ut.set_random_seed(cfg.random_seed)
        train(0, cfg)
    else:
        # evaluation
        evaluate(cfg)

if __name__ == '__main__':
    main(cfg)
    log.info("Congraduation! The Prgram runing finish. ^_^ ")