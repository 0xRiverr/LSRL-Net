import argparse
import logging
import os
import random
import shutil
import sys
import time
from statistics import mean

from medpy import metric
import numpy as np
from copy import deepcopy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import ImageFilter
from tensorboardX import SummaryWriter
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from dataloaders import utils
from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler, WeakStrongAugment)
from networks.net_factory import net_factory
from utils import losses, metrics, ramps
from val_2D import test_single_volume, test_single_volume_catseg
from scipy.stats import entropy
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='/home/nobodyjiang/datasets/ACDC/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='ACDC_wsl/AL_LS_0424', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--max_iterations', type=int,

                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list, default=[256, 256],
                    help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=4,
                    help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=3,
                    help='labeled data')
parser.add_argument('--final_labeled_num', type=int, default=7,
                    help='labeled data')

# costs
parser.add_argument('--ema_decay', type=float, default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
parser.add_argument('--use_block_dice_loss', action='store_true',
                    default=False, help='use_block_dice_loss')
parser.add_argument('--block_num', type=int, default=4,
                    help='block_num')
parser.add_argument('--model2_inchns', type=int, default=5,
                    help='model2_inchns')
args = parser.parse_args()


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 34, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    elif "Prostate":
        ref_dict = {"2": 47, "4": 111, "7": 191,
                    "11": 306, "14": 391, "18": 478, "35": 940}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def get_current_consistency_weight_T(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup_t(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def get_max_and_else(outputs_soft1):
    # predicted_classes1 = torch.argmax(outputs_soft1, dim=1)
    predicted_classes1 = torch.argmin(outputs_soft1, dim=1)
    # print(predicted_classes1.size()) # [12,256,256]

    # 创建掩码区分最大概率类别和其他类别
    # 使用 one_hot 将 predicted_classes 转换为 one-hot 编码，形状为 (B, C, H, W)
    mask1 = F.one_hot(predicted_classes1, num_classes=4).permute(0, 3, 1, 2).float()
    # max_probs = torch.gather(outputs_soft1, 1, predicted_classes1.unsqueeze(1)).squeeze(1)
    # print(mask1)
    # 处理最大概率类别的区域和其他类别的区域
    max_prob_class_areas1 = outputs_soft1 * mask1
    else_max_prob_class_areas1 = outputs_soft1 * (1 - mask1)
    # print('='*50)
    # torch.Size([12, 4, 256, 256])
    # print(max_prob_class_areas1.shape)
    # print(else_max_prob_class_areas1.shape)
    # print((max_prob_class_areas1.sum(dim=1))==(1-else_max_prob_class_areas1.sum(dim=1)))

    return max_prob_class_areas1, else_max_prob_class_areas1

def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False, in_chns=1):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=in_chns,
                            class_num=num_classes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model1 = create_model()
    model2 = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([
        WeakStrongAugment(args.patch_size)
    ]))
    # db_levelset = BaseDataSets(base_dir=args.root_path, split="mask_levelset")
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    total_slices = len(db_train)
    initial_labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    final_labeled_slice = patients_to_slices(args.root_path, args.final_labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, initial_labeled_slice))
    labeled_idxs = list(range(0, initial_labeled_slice))
    unlabeled_idxs = list(range(initial_labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    # levelsetloader = DataLoader(db_levelset, batch_sampler=batch_sampler,
    #                             num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False,
                           num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr,
                           momentum=0.9, weight_decay=0.0001)
    # optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    ce_loss1 = CrossEntropyLoss(reduction='none')
    if args.use_block_dice_loss:
        block_loss = losses.Block_DiceLoss(num_classes, args.block_num)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance1 = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        count = 0
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            volume_batch_strong, label_batch_strong = sampled_batch['image_strong'], sampled_batch['label_strong']
            volume_batch_strong, label_batch_strong = volume_batch_strong.cuda(), label_batch_strong.cuda()

            mask_levelset = sampled_batch['mask_levelset']
            mask_levelset = mask_levelset.cuda()

            # outputs1.size = torch.Size([24, 4, 256, 256]) B, C, H, W
            outputs1 = model1(volume_batch)  # 0-11为lable   12-23为unlabel
            outputs1_unlabel = outputs1[args.labeled_bs:] # 12
            outputs1_soft = torch.softmax(outputs1, dim=1)
            outputs1_unlabel_soft = torch.softmax(outputs1_unlabel, dim=1)

            with torch.no_grad():
                outputs2 = model2(volume_batch_strong)
                outputs2_unlabel = outputs2[args.labeled_bs:]
                outputs2_soft = torch.softmax(outputs2, dim=1)
                outputs2_unlabel_soft = torch.softmax(outputs2_unlabel, dim=1)

            max_prob_class_areas1, else_max_prob_class_areas1 = get_max_and_else(outputs1_unlabel_soft)
            max_prob_class_areas2, else_max_prob_class_areas2 = get_max_and_else(outputs2_unlabel_soft)

            # print(else_max_prob_class_areas2.size())  # [12, 4, 256, 256]
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            # LSup
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs], label_batch[:][:args.labeled_bs].long()) +
                               dice_loss(outputs1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))

            pseudo_outputs1 = torch.argmax(else_max_prob_class_areas1.detach(), dim=1, keepdim=False)
            pseudo_outputs2 = torch.argmax(else_max_prob_class_areas2.detach(), dim=1, keepdim=False)
            # pseudo_outputs1 = torch.argmax(outputs1_unlabel_soft.detach(), dim=1, keepdim=False)
            # pseudo_outputs2 = torch.argmax(outputs1_unlabel_soft.detach(), dim=1, keepdim=False)

            dice_similarity = []
            # print(outputs2_unlabel_soft.size())
            for i in range(args.labeled_bs):  # 0 < i < 11
                # m0 = max_prob_class_areas1[i]
                m0 = max_prob_class_areas1[i].cpu().detach().numpy()
                m1 = max_prob_class_areas2[i].cpu().detach().numpy()
                # m1 = max_prob_class_areas2[i]
                # dice_similarity.append((torch.nn.functional.kl_div(m0.log(), m1, reduction='sum')).item())
                dice_similarity.append(metric.binary.dc(m0, m1))
            # print(len(dice_similarity))
            # print(f'---------------------------------------------------------------------------------{sorted(dice_similarity)}---------------')
            # print(f'----------{max(dice_similarity)}---------------')

            good_all_unl = {}
            good_volume_unl = []
            good_label_unl = []
            # consistency_weight_TTT = get_current_consistency_weight_T(iter_num // 150)
            TTT = 0.8 + ((consistency_weight / 0.1) * 0.15)  # 阈值[0.8, 0.95]

            for i in range(args.labeled_bs):
                if dice_similarity[i] > TTT:
                    # print(dice_similarity[i])
                    good_volume_unl.append(volume_batch[args.labeled_bs:][i])
                    good_label_unl.append(label_batch[args.labeled_bs:][i])
                    good_all_unl = {'image': good_volume_unl, 'label': good_label_unl}
            # print(good_all_unl['image'])

            loss_good1 = 0
            if len(good_volume_unl) != 0:
                volume_good_unl, label_good_unl = good_all_unl['image'], good_all_unl['label']

                for i in range(len(volume_good_unl)):
                    count = count + 1
                    # print(f're-learning times: {count}')
                    if count < final_labeled_slice:
                        outputs_good1 = model1(volume_good_unl[i].unsqueeze(0))
                        outputs_good_soft1 = torch.softmax(outputs_good1, dim=1)

                        # LSup_fake_label
                        loss_good1 = loss_good1 + 0.5 * (ce_loss(outputs_good1, label_good_unl[i].unsqueeze(0).long()) +
                                                         dice_loss(outputs_good_soft1, label_good_unl[i].unsqueeze(1)))
                    else:
                        break

                if iter_num > 10000:
                    loss_good1 = 0
                else:
                    loss_good1 = loss_good1 / len(good_all_unl)



            pseudo_supervision1 = torch.mean(ce_loss1(else_max_prob_class_areas1, pseudo_outputs2) * mask_levelset[args.labeled_bs:].squeeze(1))  # pro_weak & srrong
            pseudo_supervision2 = torch.mean(ce_loss1(else_max_prob_class_areas2, pseudo_outputs1))  # weak & pro_strong & uncertainty
            #
            # pseudo_supervision1 = torch.mean(ce_loss1(outputs1_unlabel_soft, pseudo_outputs2) * mask_levelset[args.labeled_bs:].squeeze(1))  # pro_weak & srrong
            # pseudo_supervision2 = torch.mean(ce_loss1(outputs2_unlabel_soft, pseudo_outputs1))  # weak & pro_strong & uncertainty

            model1_loss = loss1 + 2 * loss_good1 + consistency_weight * (pseudo_supervision1 + pseudo_supervision2)
            # model1_loss = 0.7 * loss1 + 0.3 * loss_good1 + consistency_weight * (pseudo_supervision1 + pseudo_supervision2)

            # model2_loss = 0.7 * loss2 + 0.3 * loss_good2 + consistency_weight * pseudo_supervision2

            loss = model1_loss

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            update_ema_variables(model1, model2, args.ema_decay, iter_num)

            iter_num = iter_num + 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            logging.info('iteration %d : model1 loss : %f' % (iter_num, model1_loss.item()))
            if iter_num % 50 == 0:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                outputs = torch.argmax(torch.softmax(
                    outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction',
                                 outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0
                for i_batch, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], model1, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes - 1):
                    writer.add_scalar('info/model1_val_{}_dice'.format(class_i + 1),
                                      metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/model1_val_{}_hd95'.format(class_i + 1),
                                      metric_list[class_i, 1], iter_num)

                performance1 = np.mean(metric_list, axis=0)[0]

                mean_hd951 = np.mean(metric_list, axis=0)[1]
                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1
                    save_mode_path = os.path.join(snapshot_path,
                                                  'model1_iter_{}_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1, 4)))
                    save_best = os.path.join(snapshot_path,
                                             '{}_best_model1.pth'.format(args.model))
                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' % (iter_num, performance1, mean_hd951))
                model1.train()

            # change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
                # for param_group in optimizer2.param_groups:
                #     param_group['lr'] = lr_
            if iter_num % 3000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'model1_iter_' + str(iter_num) + '.pth')
                torch.save(model1.state_dict(), save_mode_path)
                logging.info("save model1 to {}".format(save_mode_path))

                save_mode_path = os.path.join(
                    snapshot_path, 'model2_iter_' + str(iter_num) + '.pth')
                torch.save(model2.state_dict(), save_mode_path)
                logging.info("save model2 to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
