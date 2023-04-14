#!/usr/bin/env python
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
import evaluation.eval_metric as em
from glob import glob

def calculate_EER_only(cm_score, label):
    bona_cm = cm_score[label == 1]
    spoof_cm = cm_score[label == 0]
    eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)
    tDCF = None

    return eer_cm*100,tDCF


def eval_to_score_file(score_file, cm_key_file):
    
    cm_data = pandas.read_csv(cm_key_file, sep=' ', header=None)
    submission_scores = pandas.read_csv(score_file, sep=' ', header=None, skipinitialspace=True)
    if len(submission_scores) != len(cm_data):
        print('CHECK: submission has %d of %d expected trials.' % (len(submission_scores), len(cm_data)))
        exit(1)

    # if len(submission_scores.columns) > 2:
    #     print('CHECK: submission has more columns (%d) than expected (2). Check for leading/ending blank spaces.' % len(submission_scores.columns))
    #     exit(1)
            
    # cm_scores = submission_scores.merge(cm_data[cm_data[7] == phase], left_on=1, right_on=1, how='inner')  # check here for progress vs eval set

    cm_scores = submission_scores[submission_scores[7] == phase]    # score contain meta data
    bona_cm = cm_scores[cm_scores[5] == 'bonafide'][13].values
    spoof_cm = cm_scores[cm_scores[5] == 'spoof'][13].values
    eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)
    scores = cm_scores
    spoof_sample = scores[scores[5] == 'spoof']
    bonafide_sample = scores[scores[5] == 'bonafide']
    spoof_num = len(spoof_sample)
    bonafide_num = len(bonafide_sample)
    crrent_sample = pandas.concat([spoof_sample[spoof_sample[13]<threshold],bonafide_sample[bonafide_sample[13] > threshold]],axis=0)
    crrent_number = len(spoof_sample[spoof_sample[13] < threshold]) + \
                    len(bonafide_sample[bonafide_sample[13] > threshold])
    crrent_spoof_num = len(crrent_sample[crrent_sample[5] == 'spoof'])
    crrent_bonafide_num = len(crrent_sample[crrent_sample[5] == 'bonafide'])

    # specific_attribution = {
    #     # 'neural_vocoder_autoregressive':8,
    #     'vcc2020':3,
    #     # 'high_mp3':2,
    #     'high_ogg':2,
    #     # 'high_m4a':2
    #                         }
    # crrent_specific_conbination = crrent_sample
    # for k,v in specific_attribution.items():
    #     crrent_specific_conbination = crrent_specific_conbination[crrent_specific_conbination[v]==k]
    #
    # whole_specific_conbination = scores
    # for k,v in specific_attribution.items():
    #     whole_specific_conbination = whole_specific_conbination[whole_specific_conbination[v]==k]


    subtype_acc = (crrent_number / len(scores)) * 100
    print("whole sample number: {}".format(len(scores)))
    print("whole type acc: {}".format(subtype_acc))
    print("whole type eer: {}".format(eer_cm * 100))
    print("whole type threshold: {}".format(threshold))
    print("spoof rate: {}/{} ({:.4f}), bonafide rate: {}/{} ({:.4f})".format(crrent_spoof_num, spoof_num,
                                                                   crrent_spoof_num / spoof_num * 100,
                                                                   crrent_bonafide_num, bonafide_num,
                                                                   crrent_bonafide_num / bonafide_num * 100))
    # print("conbination rate: {}/{} ({:.4f})".format(len(crrent_specific_conbination),len(whole_specific_conbination),len(crrent_specific_conbination)/len(whole_specific_conbination)*100))
    print("=============================")


    # codec = []
    # tmp = cm_scores[3].values
    # for i in range(len(tmp)):
    #     if tmp[i] not in codec:
    #         codec.append(tmp[i])
    #
    # print("{}".format(len(codec)))
    # return 0

    check = False
    if check:
        codec = ['nocodec','low_m4a','high_m4a','low_mp3','high_mp3','high_ogg','low_ogg','mp3m4a','oggm4a']
        # codec = ['asvspoof','vcc2018','vcc2020']
        codec_meta = {}
        count = 0
        for c in codec:
            codec_meta[c] = cm_scores[cm_scores[2] == c]
            scores = codec_meta[c]
            spoof_sample = scores[scores[5]=='spoof']
            bonafide_sample = scores[scores[5]=='bonafide']
            spoof_num = len(spoof_sample)
            bonafide_num = len(bonafide_sample)
            crrent_sample = pandas.concat(
                [spoof_sample[spoof_sample[13] < threshold], bonafide_sample[bonafide_sample[13] > threshold]], axis=0)
            crrent_number = len(spoof_sample[spoof_sample[13]<threshold])+ \
                            len(bonafide_sample[bonafide_sample[13]>threshold])
            crrent_spoof_num = len(crrent_sample[crrent_sample[5]=='spoof'])
            crrent_bonafide_num  = len(crrent_sample[crrent_sample[5]=='bonafide'])
            count = len(scores) + count
            subtype_acc = (crrent_number/len(scores))*100

            bona_cm = scores[scores[5] == 'bonafide'][13].values
            spoof_cm = scores[scores[5] == 'spoof'][13].values
            eer_cm, threshold = em.compute_eer(bona_cm, spoof_cm)

            print("{} sample number: {}".format(c,len(scores)))
            print("{} type acc: {}".format(c,subtype_acc))
            print("{} type eer: {}".format(c, eer_cm*100))
            print("{} type threshold: {}".format(c, threshold))
            print("spoof rate: {}/{} ({:.4f}), bonafide rate: {}/{} ({:.4f})".format(crrent_spoof_num,spoof_num,crrent_spoof_num/spoof_num*100,crrent_bonafide_num,bonafide_num,crrent_bonafide_num/bonafide_num*100))
            print("===========================")

        print("total sample number: {}".format(count))

    vocoder = False
    if vocoder:
        codec = ['traditional_vocoder', 'neural_vocoder_nonautoregressive', 'neural_vocoder_autoregressive',
                 'waveform_concatenation', 'unknown']
        codec_meta = {}
        count = 0
        for c in codec:
            codec_meta[c] = cm_scores[cm_scores[8] == c]
            scores = codec_meta[c]
            crrent_number = len(scores[scores[13] < threshold])
            count = len(scores) + count
            subtype_acc = (crrent_number / len(scores)) * 100

            print("{} sample number: {}".format(c, len(scores)))
            print("{} type acc: {}".format(c, subtype_acc))
            print("===========================")



if __name__ == "__main__":
    pass
