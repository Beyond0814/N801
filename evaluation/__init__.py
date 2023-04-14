#!/venv/Scripts/python
# -*- coding:utf-8 -*-
# @version   : V1.0
# @author    : zhongjiafeng
# @time      : 3/20/2023 5:26 PM
# @function  : the script is used to do something


# 输入为标签和预测值 输出衡量指标
import evaluation.evaluate_2019_LA as LA19
import evaluation.evaluate_2021_LA as LA21
from evaluation.evaluate_2021_DF import calculate_EER_only
from evaluation.eval_metric import eval_to_score_file

across_evaluation = {}
across_evaluation['ASVspoof19LA'] = LA19.calculate_tDCF_EER
across_evaluation['ASVspoof21LA'] = LA21.calculate_tDCF_EER
across_evaluation['ASVspoof21DF'] = calculate_EER_only
across_evaluation['In-the-Wild'] = calculate_EER_only
across_evaluation['FMFCC_A'] = calculate_EER_only


