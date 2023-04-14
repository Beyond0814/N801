import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import torch
from torch import nn
from tensorboardX import SummaryWriter
import torch.multiprocessing as mp
from torch.utils.data import DistributedSampler, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import destroy_process_group


import utility as ut
import evaluation as ev
import numpy as np
import data_util as dt

import hydra
from omegaconf import DictConfig, OmegaConf
import logging
################   custom import   ################
import data_util.asvspoof as Data
import model.wav2vec as Model

log = logging.getLogger(__name__)

def train(rank, num_gpu, cfg: DictConfig):
    if num_gpu > 1:
        ut.ddp_setup(rank,num_gpu)

    if cfg.reproducible.enable:
        ut.set_random_seed(cfg)
    device = 'cuda:{}'.format(rank)

    model_config = cfg.model_configuration
    train_config = cfg.training
    data_config = cfg.database

    data_config.protocol['train'] = data_config.protocol_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    data_config.protocol['dev'] = data_config.protocol_path + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'

    train_set = Data.get_dataset(data_config, type='train')
    train_sampler = DistributedSampler(train_set)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=train_config.batch_size,
                              shuffle=False,
                              sampler=train_sampler,
                              num_workers=train_config.num_workers
                              )


    model = Model.model(model_config,device).to(rank)
    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    if rank == 0:
        sw = SummaryWriter(os.getcwd())
        nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
        log.info('nb_params: {}'.format(nb_params))
        log.info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    for epoch in range(train_config.max_num_epoch):
        model.train()
        train_num_total = 0.0
        dev_num_total = 0.0
        train_loss = 0.0
        dev_loss = 0.0
        best_dev_loss = 99

        train_sampler.set_epoch(epoch)
        weight = torch.FloatTensor([0.1, 0.9]).to(rank)
        criterion = nn.CrossEntropyLoss(weight=weight)

        for batch_x, batch_y in train_loader:
            batch_size = batch_x.size(0)
            train_num_total += batch_size

            batch_x = batch_x.to(rank)
            batch_y = batch_y.to(rank)
            batch_output = model(batch_x)

            batch_loss = criterion(batch_output, batch_y)

            train_loss += (batch_loss.item() * batch_size)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss = train_loss / train_num_total

        if rank == 0:
            sw.add_scalar("train_loss", train_loss ,epoch)
            log.info('epoch {} , train loss: {} , dev loss: {}.'.format(epoch, train_loss, dev_loss))

    if rank== 0:
        sw.close()
    destroy_process_group()

def evaluate(cfg):
    eval_config = cfg.across_evaluate
    EER = {}
    tDCF = {}

    for data_name in eval_config.enable_dataset:
        assert  data_name in ['ASVspoof19LA','ASVspoof21LA','ASVspoof21DF','In-the-Wild'], 'Invalid evaluation set name.'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = cfg.model_configuration
        model = Model.model(model_config, device)
        model = model.to(device)
        if model_config.load_checkpoint:
            model.load_state_dict(torch.load(model_config.model_path, map_location=device))
            print('Model loaded : {}'.format(model_config.model_path))

        eval_loader = dt.get_across_eval_dataloader(data_name, cfg)
        cm_score = []
        label = []
        for batch_x, batch_y in eval_loader:
            batch_x = batch_x.to(device)
            batch_output = model(batch_x)

            cm_score.append(batch_output.data.cpu().numpy())
            label.append(batch_y)

        cm_score = np.array(cm_score)
        label = np.array(label)
        EER[data_name],tDCF[data_name] = ev.across_evaluation[data_name](cm_score,label)
        print('{} , EER : {}, t-DCF : {}'.format(data_name,EER[data_name],tDCF[data_name]))

    ut.save_across_result(EER, tDCF, eval_config.output_path)


#################################-----main function-----###############################
@hydra.main(config_path='./config', config_name='wav2vec_config')
def main(cfg: DictConfig):
    avalidable_gpu_num = torch.cuda.device_count()
    log.info('GPU number: {}'.format(avalidable_gpu_num))
    num_gpus = cfg.training.num_gpus
    assert num_gpus == avalidable_gpu_num
    assert cfg.mode in ['train','eval'], 'invalid mode, please check config file.'

    if cfg.mode == 'train':
        # Read configuration file
        assert cfg.database.track in ['LA','DF'], "invalid track."

        mp.spawn(train, nprocs=num_gpus, args=(num_gpus,cfg))


    else:
        evaluate(cfg)


if __name__ == '__main__':
    main()
    log.info("Congratulation! The Paragram finish.^_^!!!")