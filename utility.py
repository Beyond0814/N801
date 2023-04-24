#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 3:18 PM
# @function  : the script is used to do something

import os
import torch
import random
import numpy as np
from torch.distributed import init_process_group

def produce_file_list(base, output):
    file_name_list = os.listdir(base)
    with open(output, 'w') as fh:
        for file_name in file_name_list:
            fh.write('{}\n'.format(file_name))
    fh.close()

def cosine_annealing(step, total_steps, config):
    """Cosine Annealing for learning rate decay scheduler"""
    lr_min = config.lr_min
    lr_max = 1
    return lr_min + (lr_max -
                     lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "54321"
    init_process_group(backend="nccl",
                       rank=rank,
                       world_size=world_size,
                       )
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    return

def check_dir_exit_if_not_create(path):
    '''check_dir_exit_if_not_create(path):

        Check if path exit, if not then create it.

    :param path: direction path, string
    :return: None
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    return


def set_random_seed(cfg):
    """ set_random_seed(random_seed, args=None)

    Set the random_seed for numpy, python, and cudnn

    input
    -----
      random_seed: integer random seed
      args: argue parser
    """
    # initialization
    random_seed = cfg.random_seed
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return

def save_across_result(EER, tDCF, path):
    print('pretend to save!')

def get_full_score_file(dir_meta):
    file_list = []
    mos_dic = {}
    with open(dir_meta, 'r') as f:
        l_meta = f.readlines()

    for line in l_meta:
        mos ,utt = line.strip().split()
        file_list.append(utt)
        mos_dic[utt] = mos

    with open('scores.txt', 'w') as fh:
        for utt in file_list:
            fh.write('{} {}\n'.format(utt, mos_dic[utt]))

    print('finish.')

def audio_pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        start = random.randint(0,x_len-max_len)
        return x[start:start+max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x
def produce_scores_file(cm_score, file_list, save_path):
    '''
        将模型预测结果保存为score文件。

    :param cm_score: 模型的预测结果，格式为np.array
    :param file_list: 样本名，格式为np.array
    :param save_path: 分数文件存储路径
    :return:
    '''
    with open(save_path, 'w+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {}\n'.format(utt, s))
    fh.close()
    print('Scores saved to : {}'.format(save_path))

def produce_probability_file(cm_score, file_list, save_path):
    '''
        将模型预测结果保存为score文件。

    :param cm_score: 模型的预测结果，格式为np.array
    :param file_list: 样本名，格式为np.array
    :param save_path: 分数文件存储路径
    :return:
    '''

    with open(save_path, 'w+') as fh:
        for s, utt in zip(cm_score, file_list):
            fh.write('{} {} {}\n'.format(utt, s[0], s[1]))
    fh.close()
    print('Scores saved to : {}'.format(save_path))

if __name__ == '__main__':
    get_full_score_file('/home/zhongjiafeng/Model/N801/database/ADD2023-key/mos_label/eval_mos.txt')

