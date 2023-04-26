#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 4/18/2023 3:11 PM
# @function  : the script is used to do something
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5'

import torch
from torch.nn.parallel import DataParallel
import logging
import tent
from tqdm import tqdm

from omegaconf import OmegaConf
import utility_tool.utility as ut

# ===============================
import dataset_tool.fmfcc as data
import models.collection.aasist.wav2vec_aasist as Model
# ===============================
logger = logging.getLogger(__name__)
cfg = OmegaConf.load('./config.yaml')

def run_tent():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    base_model = Model.model(cfg,device)
    base_model = base_model.to(device)
    base_model.load_state_dict(torch.load(cfg.ckp_path, map_location=device))
    if cfg.num_gpus > 1:
        base_model = DataParallel(base_model).to(device)

    model = setup_tent(base_model)
    eval_loader = data.get_FMFCC_dataloader(cfg['FMFCC_A'],cfg)


    all_output = []
    all_label = []
    all_probability_output = []


    for batch_x, batch_y in tqdm(eval_loader):
        batch_x = batch_x.to(device)
        batch_out = model(batch_x)

        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        batch_all_output = batch_out.data.cpu().numpy()

        all_probability_output.extend(batch_all_output)
        all_output.extend(batch_score.tolist())
        all_label.extend(batch_y)


    print('init entropy: {} , final entropy : {} '.format(model.init_entropy,model.final_entropy))
    ut.produce_scores_file(all_output, all_label, 'tent_step_{}_scores.txt'.format(cfg.OPTIM.STEPS))
    ut.produce_probability_file(all_probability_output, all_label, 'tent_step_{}_probability.txt'.format(cfg.OPTIM.STEPS))
def setup_tent(model):
    """Set up tent adaptation.

        Configure the model for training + feature modulation by batch statistics,
        collect the parameters for feature modulation by gradient optimization,
        set up the optimizer, and then tent the model.
        """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return torch.optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return torch.optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=cfg.OPTIM.MOMENTUM,
                   dampening=cfg.OPTIM.DAMPENING,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=cfg.OPTIM.NESTEROV)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    run_tent()
    print('------------------- Finish -------------------')