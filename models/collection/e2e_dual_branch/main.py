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
from model import setup_seed, ResNet, TypeClassifier
################   custom    ################

log = logging.getLogger(__name__)


def train(rank, cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_configuration
    train_config = cfg.training
    data_config = cfg.database


    data_config.protocol['train'] = data_config.protocol_path + 'ASVspoof2019.LA.cm.train.trn.txt'
    data_config.protocol['dev'] = data_config.protocol_path + 'ASVspoof2019.LA.cm.dev.trl.txt'

    # TODO: need to redefine dataloader
    train_loader = Data.get_dataloader(data_config, train_config, type='train')
    dev_loader = Data.get_dataloader(data_config, train_config, type='dev')

    resnet_lfcc = ResNet(3, cfg.enc_dim, resnet_type='18', nclasses=2).to(device)
    resnet_cqt = ResNet(4, cfg.enc_dim, resnet_type='18', nclasses=2).to(device)
    classifier_lfcc = TypeClassifier(cfg.enc_dim, 6, cfg.lambda_, ADV=True).to(device)
    classifier_cqt = TypeClassifier(cfg.enc_dim, 6, cfg.lambda_, ADV=True).to(device)

    resnet_lfcc_optimizer = torch.optim.Adam(resnet_lfcc.parameters(), lr=cfg.lr, betas=(cfg.beta_1, cfg.beta_2),
                                             eps=cfg.eps, weight_decay=1e-4)
    resnet_cqt_optimizer = torch.optim.Adam(resnet_cqt.parameters(), lr=cfg.lr, betas=(cfg.beta_1, cfg.beta_2),
                                            eps=cfg.eps, weight_decay=1e-4)
    classifier_lfcc_optimizer = torch.optim.Adam(classifier_lfcc.parameters(), lr=cfg.lr,
                                                 betas=(cfg.beta_1, cfg.beta_2), eps=cfg.eps, weight_decay=1e-4)
    classifier_cqt_optimizer = torch.optim.Adam(classifier_cqt.parameters(), lr=cfg.lr,
                                                betas=(cfg.beta_1, cfg.beta_2), eps=cfg.eps, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(os.getcwd())

    for epoch in range(train_config.max_num_epoch):
        resnet_lfcc.train()
        resnet_cqt.train()
        classifier_lfcc.train()
        classifier_cqt.train()

        epoch_loss = []
        epoch_lfcc_ftcloss = []
        epoch_lfcc_fcloss = []
        epoch_cqt_ftcloss = []
        epoch_cqt_fcloss = []

        for i, (lfcc, cqt, label, fakelabel) in enumerate(tqdm(train_loader)):
            lfcc = lfcc.unsqueeze(1).float().to(device)
            cqt = cqt.unsqueeze(1).float().to(device)
            label = label.to(device)
            fakelabel = fakelabel.to(device)

            # get fake features and forgery type label
            feature_lfcc, out_lfcc = resnet_lfcc(lfcc)
            feature_fake_lfcc, fakelabel_lfcc = getFakeFeature(feature_lfcc, fakelabel)

            # calculate ftcloss
            typepred_lfcc = classifier_lfcc(feature_fake_lfcc)
            typeloss_lfcc = criterion(typepred_lfcc, fakelabel_lfcc)

            # optimize FTCM
            classifier_lfcc_optimizer.zero_grad()
            typeloss_lfcc.backward(retain_graph=True)
            classifier_lfcc_optimizer.step()

            # get new ftcloss
            type_pred_lfcc = classifier_lfcc(feature_fake_lfcc)
            ftcloss_lfcc = criterion(type_pred_lfcc, fakelabel_lfcc)

            # calculate fcloss
            fcloss_lfcc = criterion(out_lfcc, label)

            # cqt branch
            # get fake features and forgery type label
            feature_cqt, out_cqt = resnet_cqt(cqt)
            feature_fake_cqt, fakelabel_cqt = getFakeFeature(feature_cqt, fakelabel)

            # calculate ftcloss
            typepred_cqt = classifier_cqt(feature_fake_cqt)
            typeloss_cqt = criterion(typepred_cqt, fakelabel_cqt)

            # optimize FTCM
            classifier_cqt_optimizer.zero_grad()
            typeloss_cqt.backward(retain_graph=True)
            classifier_cqt_optimizer.step()

            # get new ftcloss
            type_pred_cqt = classifier_cqt(feature_fake_cqt)
            ftcloss_cqt = criterion(type_pred_cqt, fakelabel_cqt)

            # calculate fcloss
            fcloss_cqt = criterion(out_cqt, label)

            # LOSS
            loss = ftcloss_lfcc + fcloss_lfcc + ftcloss_cqt + fcloss_cqt
            epoch_loss.append(loss.item())

            epoch_lfcc_ftcloss.append(ftcloss_lfcc.item())
            epoch_lfcc_fcloss.append(fcloss_lfcc.item())
            epoch_cqt_ftcloss.append(ftcloss_cqt.item())
            epoch_cqt_fcloss.append(fcloss_cqt.item())

            # opyimize Feature Extraction Module and Forgery Classification Module
            resnet_lfcc_optimizer.zero_grad()
            resnet_cqt_optimizer.zero_grad()
            loss.backward()
            resnet_lfcc_optimizer.step()
            resnet_cqt_optimizer.step()


        # log something
        resnet_lfcc.eval()
        resnet_cqt.eval()
        classifier_cqt.eval()
        classifier_lfcc.eval()

        with torch.no_grad():
            dev_loss = []
            label_list = []
            scores_list = []

            for i, (lfcc, cqt, label, _) in enumerate(tqdm(valDataLoader)):
                lfcc = lfcc.unsqueeze(1).float().to(args.device)
                cqt = cqt.unsqueeze(1).float().to(args.device)
                label = label.to(args.device)

                _, out_lfcc = resnet_lfcc(lfcc)
                fcloss_lfcc = criterion(out_lfcc, label)
                score_lfcc = F.softmax(out_lfcc, dim=1)[:, 0]

                _, out_cqt = resnet_cqt(cqt)
                fcloss_cqt = criterion(out_cqt, label)
                score_cqt = F.softmax(out_cqt, dim=1)[:, 0]

                score = torch.add(score_lfcc, score_cqt)
                score = torch.div(score, 2)

                loss = fcloss_lfcc + fcloss_cqt
                dev_loss.append(loss.item())

                label_list.append(label)
                scores_list.append(score)

            scores = torch.cat(scores_list, 0).data.cpu().numpy()
            labels = torch.cat(label_list, 0).data.cpu().numpy()

            # log something
            pass
    # final save
    path = os.path.join(os.getcwd(), 'Final_model.pth')
    torch.save(model.module.state_dict(), path)

def getFakeFeature(feature,label):
    f = []
    l = []
    for i in range(0,label.shape[0]):
        if label[i]!=20:
            l.append(label[i])
            f.append(feature[i])
    f = torch.stack(f)
    l = torch.stack(l)
    return f,l

def evaluate(cfg):
    # evaluate mode is running on DP mode, DDP is not support.
    eval_config = cfg.across_evaluate

    for data_name in eval_config.enable_dataset:
        assert  data_name in ['ASVspoof19LA','ASVspoof21LA','ASVspoof21DF','In-the-Wild','FMFCC_A','toy'], \
            'Invalid evaluation set name.'


        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_config = cfg.model_configuration

        ################   custom    ################
        model = Model.model(model_config)
        model = model.to(device)
        ################   custom    ################

        if cfg.training.num_gpus > 1:
            model = DataParallel(model).to(device)
        if model_config.load_checkpoint:
            model.module.load_state_dict(torch.load(model_config.model_path, map_location=device))
            log.info('Model loaded : {}'.format(model_config.model_path))



        model.eval()
        eval_loader = dt.get_across_eval_dataloader(data_name, cfg)
        save_path = os.path.join(os.getcwd(),
                                 '{}_{}_score_file.txt'.format(data_name, cfg.model_configuration.model_name))
        cm_score = []
        utt_list = []
        with torch.no_grad():
            for batch_x, batch_utt in tqdm(eval_loader, desc=data_name, ncols=100):
                batch_x = batch_x.to(device)
                batch_output = model(batch_x)
                batch_score = (batch_output[:, 1]).data.cpu().numpy().ravel()

                cm_score.extend(batch_score.tolist())
                utt_list.extend(batch_utt)

            cm_score = np.array(cm_score).ravel()
            utt_list = np.array(utt_list)

            dt.produce_across_evaluation_file(data_name,cm_score,utt_list,
                                          cfg.all_dataset_config[data_name].protocol,
                                          save_path)
            eer,acc = ev.eval_to_score_file(save_path, log)
            log.info('{} - | EER : {} | ACC : {}'.format(data_name, eer,acc))



#################################-----main function-----###############################
@hydra.main(config_path='./config', config_name='me_config')
def main(cfg: DictConfig):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.visible_gpu
    available_gpu_num = torch.cuda.device_count()
    assert available_gpu_num == cfg.training.num_gpus,'cfg.training.num_gpus not equal with available_gpu_num.'
    log.info('GPU number: {}'.format(available_gpu_num))

    assert cfg.mode in ['train','eval'], 'invalid mode, please check config file.'
    if cfg.mode == 'train':
        # Read configuration file
        assert cfg.database.track in ['LA','DF'], "invalid track."

        # Enable reproducible , only influence training phase
        if cfg.reproducible.enable:
            ut.set_random_seed(cfg)

        train(0, cfg)

    else:
        # evaluation
        evaluate(cfg)


if __name__ == '__main__':
    main()
    log.info("Congraduation! The Prgram runing finish.^_^!!!")