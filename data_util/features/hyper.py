#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 2022/10/21 9:56
# @function  : the script is used to configure hyperparameter which is used during training model. Parameters
#              need to configure are as follows:
#
#       - trainSet_path : the path of audio file using to training model
#       - trainSet_protocol_path : the path of trainset protocol file
#
#       - devSet_path : the path of audio file using to training model
#       - devSet_protocol_path : the path of trainset protocol file
#
#       - evalSet_path : the path of audio file using to training model
#       - evalSet_protocol_path : the path of trainset protocol file
#
#       - root : where the whole data saved, the whole data should include trainset, devset, evalset and protocol file
#               this parameter is to make it easier to migrate programs to linux
#
#   Training config:
#       - output_dim
#       - batch_size
#       - epoch
#       - early_stop
#       - fixed_length
#
#   Dataloader config:
#       - shuffle
#       - num_work : Dataloader的线程数
#
#   Feature config:
#       - feature_name
#       - backend
#       - loss_type
#       - opt
#       - opt_param
#
#   Evaluation config
#       - model_name
#       - score_identity
#
##########################################################################
##                           特征参数配置                                 ##
##########################################################################
MFCC_param = {
    'num_ceps' : 20,                 # 返回MFCC系数的个数,默认为20
    'lifter': 0,                      # 高提升滤波，默认为0
    'hop_length': 160,                # 窗口滑动步长（单位N点），默认为512
    'win_length': 320,               # 窗口长度（单位N点），默认为n_fft点数
    'window': "hann",                 # 窗口类型
    'center': True,                   # 窗口中心对齐处理
    'pad_mode': "constant",           # 填充类型
    'n_fft': 512,                    # FFT的点数，默认为2048
    'nfilts': 20,                    # 滤波器组的通道数，默认为128
    'low_freq': None,                 # 滤波器组的最低频率
    'high_freq': None,                # 滤波器组的最高频率
    'power': 2 ,
}

GTCC_param = {
    'num_ceps' : 40,                  # 返回GTCC系数的个数
    'pre_emph' : 0,                   # 0 or 1, 预加重，默认为0
    'win_len'  : 2048,                # 窗口长度（N点数），默认为2048
    'win_hop'  : 512,                 # 滑动窗口步长，默认为512
    'win_type' : 'hamming',           # 窗口类型
    'nfilts'   : 40,                 # 滤波器组通道数，默认为24
    'nfft'     : 2048,                # FFT点数，默认为2048
    'low_freq' : 0,                   # 滤波器组的最低频率，默认为0
    'high_freq': None,                # 滤波器组的最高频率，默认为None
    'scale'    : 'constant',
}

LFCC_param = {
    'num_ceps' : 20,                 # 返回LFCC系数的个数，默认为13
    'pre_emph' : 1,                  # 是否预加重，默认为0
    'win_len'  : 320,               # 窗口长度
    'win_hop'  : 160,                # 滑动窗口步长
    'win_type' : 'hamming',
    'nfilts'   : 20,                # 滤波器组通道数
    'nfft'     : 512,               # FFT点数
    'low_freq' : 0,
    'high_freq': None,
    'scale'    : 'constant',
    'use_energy': True,
    'normalize': None,
}

LPS_param = {
    'n_fft' : 2048,
    'hop_length': 512,
    'win_length': 2048,
    'window': "hann",
    'centre': True,
    'pad_mode': "constant",
    'power': 2,
}

LFB_param = {
    'pre_emph' : 1,                  # 是否预加重，默认为0
    'win_len'  : 2048,               # 窗口长度
    'win_hop'  : 512,                # 滑动窗口步长
    'win_type' : 'hamming',
    'nfilts'   : 40,                # 滤波器组通道数
    'nfft'     : 2048,               # FFT点数
    'low_freq' : 0,
    'high_freq': None,
    'scale'    : 'constant',
}
