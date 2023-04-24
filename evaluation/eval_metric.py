import numpy as np
import sys
import pandas
from tqdm import tqdm
import torch
def eval_base_probability_file(pro_path,protocol_path,label_index=1):
    """
        compute acc base on probability file and protocol file
    :param pro_path:
    :param protocol_path:
    :param label_index:
    :return:
    """
    utt_2_label = {}
    label_list = []
    num_correct = 0
    with open(protocol_path,'r') as f:
        protocol_file = f.readlines()

    for line in protocol_file:
        per_line = line.strip().split(',')
        utt_2_label[per_line[0]] = per_line[label_index]

    probability_file = np.genfromtxt(pro_path, dtype=str)
    prob_array = probability_file[:,1:].astype(np.float64)

    for utt in probability_file[:,0]:
        label_list.append(float(utt_2_label[utt]))

    label_array = np.array(label_list)
    acc = compute_acc(prob_array,label_array)
    return acc


def compute_acc(scores, label):
    """
        computer acc base model output and label
    :param scores: ndarray, dimension should be (sample_num, 2)
    :param label: ndarray, dimension should be (sanmple_num, 1)
    :return: acc
    """
    pred = scores.argmax(axis=1)
    num_total = len(label)
    num_correct = (pred == label).sum().item()
    acc = 100 * (num_correct / num_total)
    return acc
def eval_base_protocol_and_score_file(score_path,protocol_path,score_index=1,label_index=1):
    '''
        computer eer and acc base on two file : scores.txt and protocol.txt
        scores.txt format :  [sample_name] .... [score]
        the score index should correspond param score_index
        protocol.txt format ： [sample_name] .... [label]
        the label index should correspond param label_index
        the fake label should be denoted by '0'
    :param score_path:  str,score.txt path
    :param protocol_path:  str, protocol.txt path
    :param score_index:  the score column index in scores.txt
    :param label_index:  the label column index in protocol.txt
    :return: eer and acc
    '''
    utt_2_label = {}
    bona_cm = []
    spoof_cm = []
    with open(protocol_path,'r') as f:
        protocol_file = f.readlines()

    for line in protocol_file:
        per_line = line.strip().split(',')
        utt_2_label[per_line[0]] = per_line[label_index]

    score_file = np.genfromtxt(score_path, dtype=str)
    assert len(score_file) == len(protocol_file)

    for line in score_file:
        if utt_2_label[line[0]] == '0':
            spoof_cm.append(line[score_index])
        else:
            bona_cm.append(line[score_index])
    bona_cm = np.array(bona_cm).astype(np.float64)
    spoof_cm = np.array(spoof_cm).astype(np.float64)

    eer_cm, threshold = compute_eer(bona_cm, spoof_cm)
    return eer_cm,threshold.item()

def eval_base_score_file(score_path):
    '''
        computer the eer and acc base on scores.txt file,
        the scores.txt format should be:
        [sample score] [label]
        label should be: str '0' or '1'

    :param score_path: scores.txt path
    :return: eer and acc
    '''
    meta = np.genfromtxt(score_path, dtype=str)
    scores = meta[:,0].astype(np.float64)
    label = meta[:,1]
    bona_cm = scores[label == '1']
    spoof_cm = scores[label == '0']
    eer_cm, threshold = compute_eer(bona_cm, spoof_cm)
    return eer_cm

def obtain_asv_error_rates(tar_asv, non_asv, spoof_asv, asv_threshold):

    # False alarm and miss rates for ASV
    Pfa_asv = sum(non_asv >= asv_threshold) / non_asv.size
    Pmiss_asv = sum(tar_asv < asv_threshold) / tar_asv.size

    # Rate of rejecting spoofs in ASV
    if spoof_asv.size == 0:
        Pmiss_spoof_asv = None
        Pfa_spoof_asv = None
    else:
        Pmiss_spoof_asv = np.sum(spoof_asv < asv_threshold) / spoof_asv.size
        Pfa_spoof_asv = np.sum(spoof_asv >= asv_threshold) / spoof_asv.size

    return Pfa_asv, Pmiss_asv, Pmiss_spoof_asv, Pfa_spoof_asv


def compute_det_curve(target_scores, nontarget_scores):

    n_scores = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    # Sort labels based on scores
    indices = np.argsort(all_scores, kind='mergesort')
    labels = labels[indices]

    # Compute false rejection and false acceptance rates
    # 注意： 默认设置为预测分数比阈值大，则认为是目标样本，预测分数比阈值小，则认为是非目标样本
    tar_trial_sums = np.cumsum(labels)
    # tar_trial_sums数组的第i个元素的值表示为：选取比第i大的分数作为阈值时，比该分数小的样本数，即也是目标样本被错误分类的数量
    nontarget_trial_sums = nontarget_scores.size - (np.arange(1, n_scores + 1) - tar_trial_sums)
    # 同理，nontarget_trial_sums表示选取第i大的分数作为阈值时，比该分数大的样本数，即也是非目标样本被错误分类的数量
    # 注意： 当选取比第i大的分数作为阈值时，这个分数对应的样本也视为比阈值小
    frr = np.concatenate((np.atleast_1d(0), tar_trial_sums / target_scores.size))  # false rejection rates
    far = np.concatenate((np.atleast_1d(1), nontarget_trial_sums / nontarget_scores.size))  # false acceptance rates
    thresholds = np.concatenate((np.atleast_1d(all_scores[indices[0]] - 0.001), all_scores[indices]))  # Thresholds are the sorted scores
    # 这里添加拼接的第一部分为了与FRR为0，FAR为1的情况对应
    return frr, far, thresholds



def compute_eer(target_scores, nontarget_scores):
    """ Returns equal error rate (EER) and the corresponding threshold. """
    frr, far, thresholds = compute_det_curve(target_scores, nontarget_scores)
    abs_diffs = np.abs(frr - far)
    min_index = np.argmin(abs_diffs)   # 找当FAR和FRR最靠近的情况
    eer = np.mean((frr[min_index], far[min_index]))   # 在FAR和FRR之间使用线性插值求相等的EER
    return eer, thresholds[min_index]


def compute_tDCF(bonafide_score_cm, spoof_score_cm, Pfa_asv, Pmiss_asv, Pfa_spoof_asv, cost_model, print_cost):
    """
    Compute Tandem Detection Cost Function (t-DCF) [1] for a fixed ASV system.
    In brief, t-DCF returns a detection cost of a cascaded system of this form,
      Speech waveform -> [CM] -> [ASV] -> decision
    where CM stands for countermeasure and ASV for automatic speaker
    verification. The CM is therefore used as a 'gate' to decided whether or
    not the input speech sample should be passed onwards to the ASV system.
    Generally, both CM and ASV can do detection errors. Not all those errors
    are necessarily equally cost, and not all types of users are necessarily
    equally likely. The tandem t-DCF gives a principled with to compare
    different spoofing countermeasures under a detection cost function
    framework that takes that information into account.
    INPUTS:
      bonafide_score_cm   A vector of POSITIVE CLASS (bona fide or human)
                          detection scores obtained by executing a spoofing
                          countermeasure (CM) on some positive evaluation trials.
                          trial represents a bona fide case.
      spoof_score_cm      A vector of NEGATIVE CLASS (spoofing attack)
                          detection scores obtained by executing a spoofing
                          CM on some negative evaluation trials.
      Pfa_asv             False alarm (false acceptance) rate of the ASV
                          system that is evaluated in tandem with the CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_asv           Miss (false rejection) rate of the ASV system that
                          is evaluated in tandem with the spoofing CM.
                          Assumed to be in fractions, not percentages.
      Pmiss_spoof_asv     Miss rate of spoof samples of the ASV system that
                          is evaluated in tandem with the spoofing CM. That
                          is, the fraction of spoof samples that were
                          rejected by the ASV system.
      cost_model          A struct that contains the parameters of t-DCF,
                          with the following fields.
                          Ptar        Prior probability of target speaker.
                          Pnon        Prior probability of nontarget speaker (zero-effort impostor)
                          Psoof       Prior probability of spoofing attack.
                          Cmiss       Cost of tandem system falsely rejecting target speaker.
                          Cfa         Cost of tandem system falsely accepting nontarget speaker.
                          Cfa_spoof   Cost of tandem system falsely accepting spoof.
      print_cost          Print a summary of the cost parameters and the
                          implied t-DCF cost function?
    OUTPUTS:
      tDCF_norm           Normalized t-DCF curve across the different CM
                          system operating points; see [2] for more details.
                          Normalized t-DCF > 1 indicates a useless
                          countermeasure (as the tandem system would do
                          better without it). min(tDCF_norm) will be the
                          minimum t-DCF used in ASVspoof 2019 [2].
      CM_thresholds       Vector of same size as tDCF_norm corresponding to
                          the CM threshold (operating point).
    NOTE:
    o     In relative terms, higher detection scores values are assumed to
          indicate stronger support for the bona fide hypothesis.
    o     You should provide real-valued soft scores, NOT hard decisions. The
          recommendation is that the scores are log-likelihood ratios (LLRs)
          from a bonafide-vs-spoof hypothesis based on some statistical model.
          This, however, is NOT required. The scores can have arbitrary range
          and scaling.
    o     Pfa_asv, Pmiss_asv, Pmiss_spoof_asv are in fractions, not percentages.
    References:
      [1] T. Kinnunen, H. Delgado, N. Evans,K.-A. Lee, V. Vestman, 
          A. Nautsch, M. Todisco, X. Wang, M. Sahidullah, J. Yamagishi, 
          and D.-A. Reynolds, "Tandem Assessment of Spoofing Countermeasures
          and Automatic Speaker Verification: Fundamentals," IEEE/ACM Transaction on
          Audio, Speech and Language Processing (TASLP).
      [2] ASVspoof 2019 challenge evaluation plan
          https://www.asvspoof.org/asvspoof2019/asvspoof2019_evaluation_plan.pdf
    """


    # Sanity check of cost parameters
    if cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0 or \
            cost_model['Cfa'] < 0 or cost_model['Cmiss'] < 0:
        print('WARNING: Usually the cost values should be positive!')

    if cost_model['Ptar'] < 0 or cost_model['Pnon'] < 0 or cost_model['Pspoof'] < 0 or \
            np.abs(cost_model['Ptar'] + cost_model['Pnon'] + cost_model['Pspoof'] - 1) > 1e-10:
        sys.exit('ERROR: Your prior probabilities should be positive and sum up to one.')

    # Unless we evaluate worst-case model, we need to have some spoof tests against asv
    if Pfa_spoof_asv is None:
        sys.exit('ERROR: you should provide false alarm rate of spoof tests against your ASV system.')

    # Sanity check of scores
    combined_scores = np.concatenate((bonafide_score_cm, spoof_score_cm))
    if np.isnan(combined_scores).any() or np.isinf(combined_scores).any():
        sys.exit('ERROR: Your scores contain nan or inf.')

    # Sanity check that inputs are scores and not decisions
    n_uniq = np.unique(combined_scores).size
    if n_uniq < 3:
        sys.exit('ERROR: You should provide soft CM scores - not binary decisions')

    # Obtain miss and false alarm rates of CM
    Pmiss_cm, Pfa_cm, CM_thresholds = compute_det_curve(bonafide_score_cm, spoof_score_cm)

    # Constants - see ASVspoof 2019 evaluation plan

    C0 = cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon']*cost_model['Cfa']*Pfa_asv
    C1 = cost_model['Ptar'] * cost_model['Cmiss'] - (cost_model['Ptar'] * cost_model['Cmiss'] * Pmiss_asv + cost_model['Pnon'] * cost_model['Cfa'] * Pfa_asv)
    C2 = cost_model['Pspoof'] * cost_model['Cfa_spoof'] * Pfa_spoof_asv;


    # Sanity check of the weights
    if C0 < 0 or C1 < 0 or C2 < 0:
        sys.exit('You should never see this error but I cannot evalute tDCF with negative weights - please check whether your ASV error rates are correctly computed?')

    # Obtain t-DCF curve for all thresholds
    tDCF = C0 + C1 * Pmiss_cm + C2 * Pfa_cm

    # Obtain default t-DCF
    tDCF_default = C0 + np.minimum(C1, C2)

    # Normalized t-DCF
    tDCF_norm = tDCF / tDCF_default

    # Everything should be fine if reaching here.
    if print_cost:

        print('t-DCF evaluation from [Nbona={}, Nspoof={}] trials\n'.format(bonafide_score_cm.size, spoof_score_cm.size))
        print('t-DCF MODEL')
        print('   Ptar         = {:8.5f} (Prior probability of target user)'.format(cost_model['Ptar']))
        print('   Pnon         = {:8.5f} (Prior probability of nontarget user)'.format(cost_model['Pnon']))
        print('   Pspoof       = {:8.5f} (Prior probability of spoofing attack)'.format(cost_model['Pspoof']))
        print('   Cfa          = {:8.5f} (Cost of tandem system falsely accepting a nontarget)'.format(cost_model['Cfa']))
        print('   Cmiss        = {:8.5f} (Cost of tandem system falsely rejecting target speaker)'.format(cost_model['Cmiss']))
        print('   Cfa_spoof    = {:8.5f} (Cost of tandem sysmte falsely accepting spoof)'.format(cost_model['Cfa_spoof']))
        print('\n   Implied normalized t-DCF function (depends on t-DCF parameters and ASV errors), t_CM=CM threshold)')
        print('   tDCF_norm(t_CM) = {:8.5f} + {:8.5f} x Pmiss_cm(t_CM) + {:8.5f} x Pfa_cm(t_CM)\n'.format(C0/tDCF_default, C1/tDCF_default, C2/tDCF_default))
        print('     * The optimum value is given by the first term (0.06273). This is the normalized t-DCF obtained with an error-free CM system.')
        print('     * The minimum normalized cost (minimum over all possible thresholds) is always <= 1.00.')
        print('')

    return tDCF_norm, CM_thresholds

def compare_scores_file(score_one,score_two,protocol):
    eer_1, threshold_1 = eval_base_protocol_and_score_file(score_one,protocol,1,1)
    eer_2, threshold_2 = eval_base_protocol_and_score_file(score_two,protocol,1,1)

    utt_to_score_1 = {}
    utt_to_score_2 = {}
    utt_list = []
    utt_to_label = {}

    with open(score_one, 'r') as f:
        score_file_1 = f.readlines()
    for line in score_file_1:
        per_line = line.strip().split()
        utt_to_score_1[per_line[0]] = float(per_line[1])

    with open(score_two, 'r') as f:
        score_file_2 = f.readlines()
    for line in score_file_2:
        per_line = line.strip().split()
        utt_to_score_2[per_line[0]] = float(per_line[1])

    with open(protocol, 'r') as f:
        protocol_file = f.readlines()
    for line in protocol_file:
        per_line = line.strip().split(',')
        utt_list.append(per_line[0])
        utt_to_label[per_line[0]] = per_line[1]


    assert len(score_file_1) == len(score_file_2)
    utt_change_to_error = []
    utt_change_to_right = []
    utt_unchange = []
    for utt in tqdm(utt_list):
        utt_s1 = utt_to_score_1[utt]
        utt_s2 = utt_to_score_2[utt]
        utt_label = utt_to_label[utt]
        if (utt_s1 < threshold_1 and utt_s2 > threshold_2):
            # predication  result change from score_1 to score_2
            if utt_label == '1':    # > threshold 视为真实语音
                utt_change_to_right.append([utt, utt_s1, utt_s2])
            else:
                utt_change_to_error.append([utt, utt_s1, utt_s2])

        elif (utt_s1 > threshold_1 and utt_s2 < threshold_2):
            if utt_label == '1':
                utt_change_to_error.append([utt, utt_s1, utt_s2])
            else:
                utt_change_to_right.append([utt, utt_s1, utt_s2])
        else:
            utt_unchange.append([utt, utt_s1, utt_s2])

    with open('score_compare_result.txt','w+') as f:
        f.write('Runing compare from {} to {}.\n'.format(score_one,score_two))
        f.write('EER 1 : {}           EER 2 : {}.\n'.format(eer_1 *100, eer_2*100))
        f.write('Threshold 1 : {}     Threshold 2 : {}.\n'.format(threshold_1, threshold_2))
        f.write('===========================================================================================\n')
        f.write('total number of change to right : {} \n'.format(len(utt_change_to_right)))
        f.write('utt change to right:    file_name      |    score 1    |    score 2    |\n')
        for line in utt_change_to_right:
            f.write('utt change to right:    {}    | {} | {} |\n'.format(line[0],line[1],line[2]))

        f.write('===========================================================================================\n')
        f.write('total number of change to error : {} \n'.format(len(utt_change_to_error)))
        f.write('utt change to error:    file_name      |    score 1    |    score 2    |\n')
        for line in utt_change_to_error:
            f.write('utt change to error:    {}    | {} | {} |\n'.format(line[0], line[1], line[2]))

        for i in range(10):
            f.write('\n')           # to easy read

        f.write('===========================================================================================\n')
        f.write('total number of unchanged         : {} \n'.format(len(utt_unchange)))
        f.write('utt unchanged       :    file_name      |    score 1    |    score 2    |\n')
        for line in utt_unchange:
            f.write('utt unchanged       :    {}    | {} | {} |\n'.format(line[0], line[1], line[2]))
        f.write('===========================================================================================\n')

    print('compare Finish.')


def turn_probability_to_score(prob_path, fun):
    utt_2_label = {}
    label_list = []
    num_correct = 0

    probability_file = np.genfromtxt(prob_path, dtype=str)
    prob_array = probability_file[:, 1:].astype(np.float64)
    score_array = fun(prob_array)    # 越大表示是正类的概率越大
    utt_array = probability_file[:, 0]


    save_path = prob_path[:-4] + '_scores.txt'
    with open(save_path,'w+') as f:
        for utt, s in zip(utt_array,score_array):
            f.write('{} {}\n'.format(utt,s))
    print('save to : {} '.format(save_path))

def softmax(intput_array):
    intput_array = torch.tensor(intput_array)
    softmax_score = torch.nn.functional.softmax(intput_array,dim=-1)
    softmax_score = np.array(softmax_score[:,1])
    return softmax_score


if __name__ == '__main__':
    score_2_path = '/home/zhongjiafeng/Model/N801/models/experiment/TT/tent/tent_step_3_scores.txt'
    score_1_path = '/home/zhongjiafeng/Model/N801/models/experiment/TT/tent/tent_step_6_scores.txt'
    protocol_path = '/home/zhongjiafeng/Model/N801/key/FMFCC-A-keys/FMFCC-key.txt'
    probal_path = '/home/zhongjiafeng/Model/N801/models/experiment/TT/tent/tent_step_6_probability.txt'
    # turn_probability_to_score(probal_path,softmax)
    eer,_  = eval_base_protocol_and_score_file(score_1_path,protocol_path,1,1)
    # # compare_scores_file(score_1_path,score_2_path,protocol_path)
    acc = eval_base_probability_file(probal_path, protocol_path)
    print(eer*100,acc)