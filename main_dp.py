import sys
import os
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
from torch.nn.parallel import DataParallel

import utility_tool.utility as ut
import evaluation.eval_metric as ev
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

################   custom    ################
import dataset_tool.asvspoof as Data
import models.collection.aasist.wav2vec_aasist as Model
cfg = OmegaConf.load('./template_config.yaml')
from utility_tool.loggers import log
################   custom    ################


def train(rank, cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_config
    train_config = cfg.train_config
    data_config = cfg.data_config

    ################   custom    ################
    train_loader = Data.get_dataloader(data_config, train_config, type='train')
    dev_loader = Data.get_dataloader(data_config, train_config, type='dev')
    model = Model.model(model_config, device)
    model = model.to(device)
    ################   custom    ################

    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    log.info('nb_params: {}'.format(nb_params))

    if model_config.load_checkpoint:
        model.load_state_dict(torch.load(model_config.model_path,map_location=device))
        log.info('Model loaded : {}'.format(model_config.model_path))

    if cfg.num_gpus > 1:
        model = DataParallel(model).to(device)

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

    for epoch in range(train_config.max_num_epoch):
        ################   TRAIN    ################
        model.train()
        train_num_total = 0.0
        train_loss = 0.0
        best_eer = 99

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
            scheduler.step()

        train_loss = train_loss / train_num_total

        ################   VLIDATION    ################
        model.eval()
        train_package = {
            'model': model,
            'dataloader': dev_loader,
            'criterion': criterion
        }
        eer,dev_loss = evaluate(cfg, train_loop=True, train_package=train_package)
        log.info('epoch [{}] :| train loss: {} | dev loss: {} | dev EER: {} |'.format(epoch, train_loss, dev_loss, eer))

        writer.add_scalar('dev_loss', dev_loss, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)

        if eer < best_eer:
            best_eer = eer
            path = os.path.join(os.getcwd(), '{}_epoch_{}_eer_model.pt'.format(epoch, best_eer))
            torch.save(model.module.state_dict(), path)

    # final save
    path = os.path.join(os.getcwd(), 'Final_model.pt')
    torch.save(model.module.state_dict(), path)

def evaluate(cfg, train_loop=False, **train_package):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if train_loop:
        model = train_package['model']
        model = model.to(device)
        eval_loader = train_package['dataloader']
        criterion =train_package['criterion']

    else:
        # evaluate mode is running on DP mode, DDP is not support.
        eval_config = cfg.evaluate
        data_config = cfg.eval_dataset
        model_config = cfg.model_config


        ################   custom    ################
        model = Model.model(model_config)
        model = model.to(device)
        ################   custom    ################

        if cfg.num_gpus > 1:
            model = DataParallel(model).to(device)
        if model_config.load_checkpoint:
            model.module.load_state_dict(torch.load(model_config.ckp_path, map_location=device))
            log.info('Model loaded : {}'.format(model_config.ckp_path))

        # TODO ： 在对应数据集上定义get_eval_dataloader函数
        # eval_loader = Data.get_eval_dataloader(eval_config, data_config)
        eval_loader = None

    model.eval()
    loss = 0
    num_total = 0
    cm_score = []
    utt_list = []
    with torch.no_grad():
        for batch_x, batch_utt in tqdm(eval_loader, desc='eval', ncols=150):
            batch_size = batch_x.size(0)
            num_total += batch_size

            batch_x = batch_x.to(device)
            batch_output = model(batch_x)
            batch_score = (batch_output[:, 1]).data.cpu().numpy().ravel()

            if train_loop:
                batch_loss = criterion(batch_output, batch_utt)
                loss += (batch_loss.item() * batch_size)

            cm_score.extend(batch_score.tolist())
            utt_list.extend(batch_utt)

    if train_loop:
        loss = loss / num_total
        cm_score = np.array(cm_score).ravel()
        label = np.array(utt_list)
        eer = ev.eval_base_numpy_array(cm_score, label)
        return eer,loss
    else:
        cm_score = np.array(cm_score).ravel()
        utt_list = np.array(utt_list)

        prefix = 'train_loop' if train_loop else 'eval'
        save_path = os.path.join(os.getcwd(), '{}_score_file.txt'.format(prefix))
        ut.produce_scores_file(cm_score, utt_list, save_path)


#################################-----main function-----###############################
def main(cfg: DictConfig):
    available_gpu_num = torch.cuda.device_count()
    log.info('GPU visible: {} - number of GPU : {}'.format(cfg.GPU_visible,available_gpu_num))
    log.info('running mode: {} '.format(cfg.mode))
    log.info('random seed : {} '.format(cfg.random_seed))

    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU_visible

    assert cfg.mode in ['train','eval'], 'invalid mode, please check config file.'
    if cfg.mode == 'train':
        ut.set_random_seed(cfg.random_seed)
        train(0, cfg)

    else:
        evaluate(cfg)

if __name__ == '__main__':
    main(cfg)
    log.info("Congraduation! The Prgram runing finish.^_^!!!")