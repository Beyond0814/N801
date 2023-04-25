#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 4/25/2023 11:24 AM
# @function  : the script is used to do something

import logging
import os
import time
import colorlog

def __logfun(isfile=False):
    # black, red, green, yellow, blue, purple, cyan(青) and white, bold(亮白色)
    log_colors_config = {
        'DEBUG': 'bold_white',
        'INFO': 'bold',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red', # 加bold后色彩变亮
    }
    logger = logging.getLogger()
    logger.setLevel(level=logging.INFO) # 某些python库文件中有一些DEBUG级的输出信息，如果这里设置为DEBUG，会导致console和log文件中写入海量信息
    console_formatter = colorlog.ColoredFormatter(
        # fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
        fmt='%(log_color)s %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        # datefmt='%Y-%m-%d  %H:%M:%S',
        log_colors=log_colors_config
    )
    console = logging.StreamHandler()  # 输出到console的handler
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    # 输出到文件
    if isfile:
        # 设置文件名
        time_line = time.strftime('%Y_%m_%d_%H_%M', time.localtime(time.time()))

        log_path=os.path.join(os.getcwd(),time_line)
        if not os.path.exists(log_path):
            os.mkdir(log_path)
        logfile = log_path + '/log.txt'
        os.chdir(log_path)

        # 设置文件日志格式
        filer = logging.FileHandler(logfile,mode='w') # 输出到log文件的handler
        # filer.setLevel(level=logging.DEBUG)
        file_formatter = logging.Formatter(
            fmt='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d  %H:%M:%S'
        )
        # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
        filer.setFormatter(file_formatter)
        logger.addHandler(filer)
    return logger

log=__logfun(True)

if __name__=='__main__':
    log.debug('This is a debug message.')
    log.info('This is an info message.')
    log.warning('This is a warning message.')
    log.error('This is an error message.')
    log.critical('This is a critical message.')
    log.info(os.getcwd())
