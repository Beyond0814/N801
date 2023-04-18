import sys
sys.path.append('../../')
import os
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

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
import dataset_tool.wavefake as Data
from model import DG_model, Discriminator
from hard_triplet_loss import HardTripletLoss
from AdLoss import Real_AdLoss, Fake_AdLoss
################   custom    ################

log = logging.getLogger(__name__)


def train(rank, cfg: DictConfig):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = cfg.model_configuration
    train_config = cfg.training
    opti_config = train_config.optimizer

    real_loader, fake_loader_1, fake_loader_2, fake_loader_3, fake_loader_4 = Data.get_dataset(train_config)

    DG_classify = DG_model(model_config.model_type).to(device)
    ad_net_real = Discriminator().to(device)

    nb_params = sum([param.view(-1).size()[0] for param in DG_classify.parameters()])

    if model_config.load_checkpoint:
        # TODO: 装载参数
        pass
        log.info('Model loaded : {}'.format(model_config.model_path))

    if cfg.training.num_gpus > 1:
        DG_classify = DataParallel(DG_classify).to(device)

    log.info('nb_params: {}'.format(nb_params))
    log.info('** start training target model! **\n')
    log.info(
        '--------|--------- VALID --------|--- classifier ---|------ Current Best ------|\n')
    log.info(
        '  iter  |   loss   top-1   EER   |   loss   top-1   |     top-1      EER       |\n')
    log.info(
        '-------------------------------------------------------------------------------|\n')
    writer = SummaryWriter(os.getcwd())


    ################   custom    ################
    criterion = {
        'softmax': nn.CrossEntropyLoss().cuda(),
        'triplet': HardTripletLoss(margin=0.1, hardest=False).cuda()
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, DG_classify.parameters()), "lr": opti_config.init_lr},
        {"params": filter(lambda p: p.requires_grad, ad_net_real.parameters()), "lr": opti_config.init_lr},
    ]
    optimizer = torch.optim.SGD(optimizer_dict, lr=opti_config.init_lr, momentum=opti_config.momentum, weight_decay=opti_config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    max_iter = train_config.max_iter
    iter_per_epoch = 10
    ################   custom    ################

    real_loader_iter = iter(real_loader)
    fake_loader_1_iter = iter(fake_loader_1)
    fake_loader_2_iter = iter(fake_loader_2)
    fake_loader_3_iter = iter(fake_loader_3)
    fake_loader_4_iter = iter(fake_loader_4)

    best_classifier_loss = 99
    best_eer = 99
    for iter_num in range(max_iter):
        DG_classify.train(True)
        ad_net_real.train(True)
        optimizer.zero_grad()

        # TODO ： 抽取不同域的样本
        real_domain_data, real_domain_label = next(real_loader_iter)
        fake_domain_1_data, fake_domain_1_label = next(fake_loader_1_iter)
        fake_domain_2_data, fake_domain_2_label = next(fake_loader_2_iter)
        fake_domain_3_data, fake_domain_3_label = next(fake_loader_3_iter)
        fake_domain_4_data, fake_domain_4_label = next(fake_loader_4_iter)

        input_data = torch.cat([real_domain_data,fake_domain_1_data,fake_domain_2_data,fake_domain_3_data,fake_domain_4_data],dim=0)
        input_label = torch.cat([real_domain_label,fake_domain_1_label,fake_domain_2_label,fake_domain_3_label,fake_domain_4_label],dim=0)

        ######### forward #########
        # TODO: 原paper在图像上做，在维度上需要进行改变，包括预训练模型的通道，原代码要求3通道，2D特征数据
        classifier_out, feature = DG_classify(input_data, train_config.norm_flag)

        ######### single side adversarial learning #########
        # TODO : 把真样本的特征抽出来
        feature_real = None
        # discriminator_out_real = ad_net_real(feature_real)

        ######### unbalanced triplet loss #########
        # TODO: 创建标签数据，用于三元损失
        source_domain_label = None
        triplet = criterion["triplet"](feature, source_domain_label)

        ######### cross-entropy loss #########
        # TODO: 鉴别器的loss 和 分类器的loss
        # real_adloss = Real_AdLoss(discriminator_out_real, criterion["softmax"], real_shape_list)
        cls_loss = criterion["softmax"](classifier_out.narrow(0, 0, input_data.size(0)), input_label)

        if best_classifier_loss<cls_loss:
            best_classifier_loss = cls_loss

        ######### backward #########
        # total_loss = cls_loss + config.lambda_triplet * triplet + config.lambda_adreal * real_adloss
        total_loss = cls_loss
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        writer.add_scalar('classifier loss', cls_loss.item(), iter_num)

        # TODO: 计算此时分类器在数据集上的正确率
        # log.info(
        #     '--------|--------- VALID --------|--- classifier ---|------ Current Best ------|\n')
        # log.info(
        #     '  iter  |   loss   top-1   EER   |   loss   top-1   |     top-1      EER       |\n')
        # log.info(
        #     '-------------------------------------------------------------------------------|\n')
        log.info(
            '  %4.1f |   %5.3f  %6.3f  %6.3f   |   %6.3f    %6.3f  |     %6.3f  %6.3f     |\n'
            .format(
                iter_num, 0, 0, 0, cls_loss, best_classifier_loss , 0, 0
            ))

        # validation
        # if iter_num != 0 and iter_num % iter_per_epoch == 0:
        #     scores_path, dev_loss = evaluate(model, dev_loader, criterion, validate_mode=True)
        #     eer, acc = ev.eval_to_score_file(scores_path, log)
        #     log.info(
        #         'epoch [{}] - TLoss : {} | DLoss : {} | EER : {} | ACC : {}'.format(epoch, train_loss, dev_loss, eer,
        #                                                                             acc))
        #
        #     writer.add_scalar('validation set loss curve', dev_loss, epoch)
        #     writer.add_scalar('train loop loss', train_loss, epoch)
        #     writer.add_scalar('learning rate', tqdm_dis['lr'], epoch)

    # final save
    # path = os.path.join(os.getcwd(), 'Final_model.pth')
    # torch.save(model.module.state_dict(), path)

def evaluate(model,dataloader,criterion,validate_mode=False):
    '''
        两个作用：
        1. 在训练中进行对验证集进行评估，产生分数文件： 要求输入dataloader返回 样本数据 和 label
        2. 在评估阶段对测试集进行评估，产生分数文件： 要求输入dataloader返回 样本数据 和 样本名
    :param model:  模型
    :param dataloader:
    :param criterion:  损失函数
    :param validate_mode:  当用于第一个目的时，设置为True，默认为False
    :return:  scores文件路径和验证集损失，当validate_mode=False时，验证集损失恒为0.
    '''
    # evaluate mode is running on DP mode, DDP is not support.

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    prefix = 'train_loop' if validate_mode else 'eval_phase'
    save_path = os.path.join(os.getcwd(),'{}_scores.txt'.format(prefix))

    model.eval()
    cm_score = []
    utt_list = []
    dev_loss = 0.0
    dev_num_total = 0
    with torch.no_grad():
        for batch_x, batch_utt in tqdm(dataloader, ncols=100):
            batch_x = batch_x.to(device)
            batch_output = model(batch_x)
            batch_score = (batch_output[:, 1]).data.cpu().numpy().ravel()

            if validate_mode:
                batch_size = batch_x.size(0)
                dev_num_total += batch_size
                batch_loss = criterion(batch_output, batch_utt)
                dev_loss += (batch_loss.item() * batch_size)

            cm_score.extend(batch_score.tolist())
            utt_list.extend(batch_utt)


        dev_loss = dev_loss / dev_num_total if validate_mode else 0
        cm_score = np.array(cm_score).ravel()
        utt_list = np.array(utt_list)

    ut.produce_scores_file(cm_score,utt_list,save_path)

    return save_path, dev_loss

#################################-----main function-----###############################
@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    available_gpu_num = torch.cuda.device_count()
    assert available_gpu_num == cfg.training.num_gpus,'cfg.training.num_gpus not equal with available_gpu_num.'
    log.info('GPU number: {}'.format(available_gpu_num))

    if cfg.mode == 'train':
        ut.set_random_seed(cfg)

        train(0, cfg)

    else:
        # evaluation
        pass


if __name__ == '__main__':
    main()
    log.info("==========================  FINISH. ==================================")