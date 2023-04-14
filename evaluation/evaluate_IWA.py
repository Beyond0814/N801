#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 2/26/2023 4:31 PM
# @function  : the script is used to do something

# !/usr/bin/env python
"""
Script to compute pooled EER for ASVspoof2021 DF.
Usage:
$: python PATH_TO_SCORE_FILE PATH_TO_GROUDTRUTH_DIR phase

 -PATH_TO_SCORE_FILE: path to the score file
 -PATH_TO_GROUNDTRUTH_DIR: path to the directory that has the CM protocol.
    Please follow README, download the key files, and use ./keys
 -phase: either progress, eval, or hidden_track
Example:
$: python evaluate.py score.txt ./keys eval
"""
import sys, os.path
import numpy as np
import pandas
import eval_metric as em

submit_file = '/home/zhongjiafeng/Model/SSL/model_CM_scores_file_SSL_IWA.txt'
cm_key_file = '/pubdata/zhongjiafeng/IWA/meta.csv'


def eval_to_score_file(score_file, cm_key_file):
    cm_data = pandas.read_csv(cm_key_file,sep=',')
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    cm_scores = submission_scores
    bona_cm = cm_scores[cm_data['label'] == 'bona-fide'][1].values
    spoof_cm = cm_scores[cm_data['label'] == 'spoof'][1].values
    eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)


    print("whole type eer: {}".format(eer_cm * 100))
    print("whole type threshold: {}".format(threshold))
    print("=============================")


if __name__ == "__main__":
    _ = eval_to_score_file(submit_file, cm_key_file)
