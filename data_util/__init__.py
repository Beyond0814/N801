#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 2:42 PM
# @function  : the script is used to do something

import data_util.asvspoof as asv
import data_util.FMFCC as fmfcc
from torch.utils.data import DataLoader
import data_util.compress as cpr
import data_util.add as add

across_eval_dataset = {}
across_eval_dataset['ASVspoof19LA'] = asv.Dataset_eval19LA
across_eval_dataset['ASVspoof21LA'] = asv.Dataset_eval21LA
across_eval_dataset['ASVspoof21DF'] = asv.Dataset_eval21DF
across_eval_dataset['FMFCC_A'] = fmfcc.Dataset_eval
across_eval_dataset['toy'] = cpr.toy_dataset
across_eval_dataset['ADD2023'] = add.Dataset_eval


def produce_across_evaluation_file(name, cm_score, file_list, protocol_dir, save_path):
    if name == 'FMFCC_A':
        fmfcc.produce_evaluation_file(cm_score, file_list, protocol_dir, save_path)
    elif name == 'toy':
        cpr.produce_evaluation_file(cm_score, file_list, protocol_dir, save_path)
    elif name == 'ADD2023':
        add.produce_evaluation_file(cm_score, file_list,save_path)
    return

def get_across_eval_dataloader(name, cfg):
    '''
        get the corresponding function with name.
    :param name:
    :param cfg:
    :return:
    '''
    assert name in ['ASVspoof19LA', 'ASVspoof21LA', 'ASVspoof21DF', 'In-the-Wild','FMFCC_A','toy','ADD2023']

    eval_dataset_config = cfg.all_dataset_config
    eval_config = cfg.training if cfg.mode=='train' else cfg.across_evaluate
    if name in ['ASVspoof19LA']:
        label_dic,file_name_list = asv.genSpoof_list(
            dir_meta=eval_dataset_config.ASVspoof19LA.protocol,is_train=True, is_eval=False)
        eval_set = across_eval_dataset[name](eval_config, file_name_list, label_dic, eval_dataset_config[name].base_dir)

    elif name in ['ASVspoof21LA', 'ASVspoof21DF']:
        label_dic, file_name_list = asv.genSpoof_list(
            dir_meta=eval_dataset_config.format.protocol,is_train=False, is_eval=False)
        eval_set = across_eval_dataset[name](eval_config, file_name_list, label_dic, eval_dataset_config[name].base_dir)

    elif name in ['In-the-Wild']:
        pass

    elif name in ['ADD2023']:
        file_name_list = add.genSpoof_list(dir_meta=eval_dataset_config[name].protocol, is_eval=True)
        eval_set = across_eval_dataset[name](eval_config, eval_dataset_config[name], file_name_list)

    elif name in ['FMFCC_A']:
        label_dic, file_name_list = fmfcc.genSpoof_list(dir_meta=eval_dataset_config[name].protocol )
        eval_set = across_eval_dataset[name](eval_config,eval_dataset_config[name] ,file_name_list, label_dic)
    elif name in ['toy']:
        label_dic, file_name_list = cpr.gen_toy_list(eval_dataset_config[name].protocol)
        eval_set = across_eval_dataset[name](config=eval_config, file_name_list=file_name_list, base_dir=eval_dataset_config[name].base_dir)


    eval_loader = DataLoader(
        eval_set,
        batch_size=eval_config.batch_size,
        num_workers=eval_config.num_workers,
        shuffle=False,
        drop_last=False
    )

    return eval_loader


