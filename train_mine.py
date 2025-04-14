from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import scipy.stats
import math
from data_loader import SYSUData, RegDBData, RegDBData_u, TestData, SYSUData_u, sysuData, sysuData_u
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model_mine import embed_net, Normalize
from utils import *
from loss import OriTripletLoss, CenterTripletLoss, CrossEntropyLabelSmooth, TripletLoss_WRT, information_entroy_loss, \
    ContrastiveLoss
from tensorboardX import SummaryWriter
from re_rank import random_walk, k_reciprocal
from loggers import Loggers, fake_label_record, centre_record, dis_calculation, dis_cos_cal, \
    fake_label_move, centre_record_all, del_process, del_process2, save_memory
from cluster_utils import DBSCAN, centre_calculate, generate_pseudo_labels, jaccard_dis_cal

from save_vis import save_features, init_memory

import numpy as np

np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=100, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

parser.add_argument('--share_net', default=2, type=int,
                    metavar='share', help='[1,2,3,4,5]the start number of shared network in the two-stream networks')
parser.add_argument('--re_rank', default='no', type=str, help='performing reranking. [random_walk | k_reciprocal | no]')
parser.add_argument('--pcb', default='on', type=str, help='performing PCB, on or off')
parser.add_argument('--w_center', default=2.0, type=float, help='the weight for center loss')

parser.add_argument('--local_feat_dim', default=256, type=int,
                    help='feature dimention of each local feature in PCB')
parser.add_argument('--num_strips', default=6, type=int,
                    help='num of local strips in PCB')

parser.add_argument('--label_smooth', default='on', type=str, help='performing label smooth or not')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)
loggers = Loggers(os.path.join(args.model_path, 'log.txt'))

dataset = args.dataset
if dataset == 'regdb':
    ave = torch.zeros(206, 256)
    ave2 = torch.zeros(206, 1536)
    memory_v = torch.zeros(1648, 1536)
    memory_t = torch.zeros(1648, 1536)
    for i in range(6):
        torch.save(ave, 'RegDB/visible_centre_' + str(i) + '.pth')
        torch.save(ave, 'RegDB/thermal_centre_' + str(i) + '.pth')

    torch.save(ave2, 'RegDB/visible_centre_all.pth')
    torch.save(ave2, 'RegDB/thermal_centre_all.pth')
elif dataset == 'sysu':
    ave = torch.zeros(296, 256)
    ave2 = torch.zeros(296, 1536)
    memory_v = torch.zeros(4000, 1536)
    memory_t = torch.zeros(4000, 1536)
    for i in range(6):
        torch.save(ave, 'SYSU-MM01/visible_centre_' + str(i) + '.pth')
        torch.save(ave, 'SYSU-MM01/thermal_centre_' + str(i) + '.pth')

    torch.save(ave2, 'SYSU-MM01/visible_centre_all.pth')
    torch.save(ave2, 'SYSU-MM01/thermal_centre_all.pth')

if dataset == 'sysu':
    data_path = "/SYSU-MM01/"
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]
elif dataset == 'regdb':
    data_path = "/RegDB/"
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]

checkpoint_path = args.model_path

init_memory(4, 0, 1)
init_memory(4, 0, 0)
init_memory(4, 6, 1)
init_memory(4, 6, 0)
init_memory(4, 42, 1)
init_memory(4, 42, 0)

init_memory(9, 0, 1)
init_memory(9, 0, 0)
init_memory(9, 6, 1)
init_memory(9, 6, 0)
init_memory(9, 42, 1)
init_memory(9, 42, 0)

init_memory(11, 0, 1)
init_memory(11, 0, 0)
init_memory(11, 6, 1)
init_memory(11, 6, 0)
init_memory(11, 42, 1)
init_memory(11, 42, 0)

init_memory(12, 0, 1)
init_memory(12, 0, 0)
init_memory(12, 6, 1)
init_memory(12, 6, 0)
init_memory(12, 42, 1)
init_memory(12, 42, 0)

init_memory(14, 0, 1)
init_memory(14, 0, 0)
init_memory(14, 6, 1)
init_memory(14, 6, 0)
init_memory(14, 42, 1)
init_memory(14, 42, 0)

init_memory(15, 0, 1)
init_memory(15, 0, 0)
init_memory(15, 6, 1)
init_memory(15, 6, 0)
init_memory(15, 42, 1)
init_memory(15, 42, 0)

init_memory(19, 0, 1)
init_memory(19, 0, 0)
init_memory(19, 6, 1)
init_memory(19, 6, 0)
init_memory(19, 42, 1)
init_memory(19, 42, 0)

init_memory(52, 0, 1)
init_memory(52, 0, 0)
init_memory(52, 6, 1)
init_memory(52, 6, 0)
init_memory(52, 42, 1)
init_memory(52, 42, 0)

init_memory(30, 0, 1)
init_memory(30, 0, 0)
init_memory(30, 6, 1)
init_memory(30, 6, 0)
init_memory(30, 42, 1)
init_memory(30, 42, 0)

init_memory(31, 0, 1)
init_memory(31, 0, 0)
init_memory(31, 6, 1)
init_memory(31, 6, 0)
init_memory(31, 42, 1)
init_memory(31, 42, 0)

init_memory(53, 0, 1)
init_memory(53, 0, 0)
init_memory(53, 6, 1)
init_memory(53, 6, 0)
init_memory(53, 42, 1)
init_memory(53, 42, 0)

init_memory(54, 0, 1)
init_memory(54, 0, 0)
init_memory(54, 6, 1)
init_memory(54, 6, 0)
init_memory(54, 42, 1)
init_memory(54, 42, 0)

init_memory(80, 0, 1)
init_memory(80, 0, 0)
init_memory(80, 6, 1)
init_memory(80, 6, 0)
init_memory(80, 42, 1)
init_memory(80, 42, 0)

init_memory(95, 0, 1)
init_memory(95, 0, 0)
init_memory(95, 6, 1)
init_memory(95, 6, 0)
init_memory(95, 42, 1)
init_memory(95, 42, 0)

init_memory(96, 0, 1)
init_memory(96, 0, 0)
init_memory(96, 6, 1)
init_memory(96, 6, 0)
init_memory(96, 42, 1)
init_memory(96, 42, 0)

init_memory(72, 0, 1)
init_memory(72, 0, 0)
init_memory(72, 6, 1)
init_memory(72, 6, 0)
init_memory(72, 42, 1)
init_memory(72, 42, 0)

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset + '_c_tri_pcb_{}_w_tri_{}'.format(args.pcb, args.w_center)
if args.pcb == 'on':
    suffix = suffix + '_s{}_f{}'.format(args.num_strips, args.local_feat_dim)

suffix = suffix + '_share_net{}'.format(args.share_net)
if args.method == 'agw':
    suffix = suffix + '_agw_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_gm10_k{}_p{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
loggers("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

print('==> Loading data..')

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':

    trainset = sysuData(data_path, transform=transform_train)
    trainset_u = sysuData_u(data_path, transform=transform_train)

    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    color_pos_u, thermal_pos_u = GenIdx(trainset_u.train_color_label_u, trainset_u.train_thermal_label_u)


    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':

    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    trainset_u = RegDBData_u(data_path, args.trial, transform=transform_train)

    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
    color_pos_u, thermal_pos_u = GenIdx(trainset_u.train_color_label_u, trainset_u.train_thermal_label_u)


    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))


gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
n_color = len(trainset.train_color_label)
n_thermal = len(trainset.train_thermal_label)
nquery = len(query_label)
ngall = len(gall_label)

loggers('Dataset {} statistics:'.format(dataset))
loggers('  ------------------------------')
loggers('  subset   | # ids | # images')
loggers('  ------------------------------')
loggers('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
loggers('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
loggers('  ------------------------------')
loggers('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
loggers('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
loggers('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb,
                    local_feat_dim=args.local_feat_dim, num_strips=args.num_strips)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch, share_net=args.share_net, pcb=args.pcb)
net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))


if args.label_smooth == 'on':
    criterion_id = nn.CrossEntropyLoss()
else:
    criterion_id = CrossEntropyLabelSmooth(n_class)

if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos

    criterion_tri = CenterTripletLoss(batch_size=loader_batch, margin=args.margin)

criterion_id.to(device)
criterion_tri.to(device)

if args.optim == 'sgd':
    if args.pcb == 'on':
        ignored_params = list(map(id, net.local_conv_list.parameters())) \
                         + list(map(id, net.fc_list.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.local_conv_list.parameters(), 'lr': args.lr},
            {'params': net.fc_list.parameters(), 'lr': args.lr}
        ],
            weight_decay=5e-4, momentum=0.9, nesterov=True)
    else:
        ignored_params = list(map(id, net.bottleneck.parameters())) \
                         + list(map(id, net.classifier.parameters()))

        base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

        optimizer = optim.SGD([
            {'params': base_params, 'lr': 0.1 * args.lr},
            {'params': net.bottleneck.parameters(), 'lr': args.lr},
            {'params': net.classifier.parameters(), 'lr': args.lr}],
            weight_decay=5e-4, momentum=0.9, nesterov=True)


def adjust_learning_rate(optimizer, epoch):

    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif 10 <= epoch < 30:
        lr = args.lr
    elif 30 <= epoch < 40:
        lr = args.lr * 0.1
    elif epoch >= 40:
        lr = args.lr * 0.05

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch, cindex_l, tindex_l, cindex_ul, tindex_ul, memory_v, memory_t):
    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    loss_func = ContrastiveLoss(batch_size=16)

    num_u = 4 * 4
    loss_den = 0

    if dataset == 'sysu':
        dataset_name = 'SYSU-MM01'
    elif dataset == 'regdb':
        dataset_name = 'RegDB'


    net.train()
    end = time.time()
    if epoch < 6:
        num = args.num_pos * args.batch_size
    else:
        num = 4 * 4

    for (batch_idx, (input1, input2, label1, label2)), (batch_idx, (input3, input4, label3, label4)) \
            in zip(enumerate(trainloader), enumerate(trainloader_u)):
        labels = torch.cat((label1, label2), 0)

        if epoch > 5:
            input1 = torch.cat((input1, input3), dim=0)
            input2 = torch.cat((input2, input4), dim=0)

        if epoch == 0:
            input3 = Variable(input3.cuda())
            input4 = Variable(input4.cuda())

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        cindex = cindex_l[batch_idx * num:(batch_idx + 1) * num]
        tindex = tindex_l[batch_idx * num:(batch_idx + 1) * num]
        cindex_u = cindex_ul[batch_idx * num_u:(batch_idx + 1) * num_u]
        tindex_u = tindex_ul[batch_idx * num_u:(batch_idx + 1) * num_u]

        ind1 = label1.shape[0]
        ind2 = label2.shape[0]
        ind3 = label3.shape[0]

        if args.pcb == 'on':
            feat, out0, feat_all = net(input1, input2)

            if epoch == 0:

                for i in range(label1.shape[0]):
                    if label1[i] == 4 or label1[i] == 9 or label1[i] == 11 or label1[i] == 12 or label1[i] == 14 or \
                            label1[i] == 15 or label1[i] == 19 or label1[i] == 52 or label1[i] == 30 or label1[i] == 31 \
                            or label1[i] == 53 or label1[i] == 54 or label1[i] == 80 or label1[i] == 95 or label1[
                        i] == 96 \
                            or label1[i] == 72:
                        if unorderd1.get(cindex[i]) is None:
                            unorderd1[cindex[i]] = 1
                            if cindex[i] < 3881:
                                save_features(feat_all[i], label1[i], epoch, 1, 0, 0)
                            else:
                                save_features(feat_all[i], label1[i], epoch, 1, 0, 1)
                for i in range(label2.shape[0]):
                    if label2[i] == 4 or label2[i] == 9 or label2[i] == 11 or label2[i] == 12 or label2[i] == 14 or \
                            label2[i] == 15 or label2[i] == 19 or label2[i] == 52 or label2[i] == 30 or label2[i] == 31 \
                            or label2[i] == 53 or label2[i] == 54 or label2[i] == 80 or label2[i] == 95 or label2[
                        i] == 96 \
                            or label2[i] == 72:
                        if unorderd2.get(tindex[i]) is None:
                            unorderd2[tindex[i]] = 1
                            if tindex[i] < 1983:
                                save_features(feat_all[i + ind1], label2[i], epoch, 0, 0, 0)
                            else:
                                save_features(feat_all[i + ind1], label2[i], epoch, 0, 0, 1)



            if epoch == 42:
                for i in range(label1.shape[0]):
                    if label1[i] == 4 or label1[i] == 9 or label1[i] == 11 or label1[i] == 12 or label1[i] == 14 or \
                            label1[i] == 15 or label1[i] == 19 or label1[i] == 52 or label1[i] == 30 or label1[i] == 31 \
                            or label1[i] == 53 or label1[i] == 54 or label1[i] == 80 or label1[i] == 95 or label1[i] == 96 \
                            or label1[i] == 72:
                        if unorderd1.get(cindex[i]) is None:
                            unorderd1[cindex[i]] = 1
                            if cindex[i] < 3881:
                                save_features(feat_all[i], label1[i], epoch, 1, 0, 0)
                            else:
                                save_features(feat_all[i], label1[i], epoch, 1, 0, 1)
                for i in range(label2.shape[0]):
                    if label2[i] == 4 or label2[i] == 9 or label2[i] == 11 or label2[i] == 12 or label2[i] == 14 or \
                            label2[i] == 15 or label2[i] == 19 or label2[i] == 52 or label2[i] == 30 or label2[i] == 31 \
                            or label2[i] == 53 or label2[i] == 54 or label2[i] == 80 or label2[i] == 95 or label2[
                        i] == 96 \
                            or label2[i] == 72:
                        if unorderd2.get(tindex[i]) is None:
                            unorderd2[tindex[i]] = 1
                            if tindex[i] < 1983:
                                save_features(feat_all[i + ind1 + ind3], label2[i], epoch, 0, 0, 0)
                            else:
                                save_features(feat_all[i + ind1 + ind3], label2[i], epoch, 0, 0, 1)
                for i in range(label3.shape[0]):
                    if label3[i] == 4 or label3[i] == 9 or label3[i] == 11 or label3[i] == 12 or label3[i] == 14 or \
                            label3[i] == 15 or label3[i] == 19 or label3[i] == 52 or label3[i] == 30 or label3[i] == 31\
                            or label3[i] == 53 or label3[i] == 54 or label3[i] == 80 or label3[i] == 95 or label3[i] == 96\
                            or label3[i] == 72:
                        if unorderd3.get(cindex_u[i]) is None:
                            unorderd3[cindex_u[i]] = 1
                            save_features(feat_all[i + ind1], label3[i], epoch, 1, 1, 4)
                for i in range(label4.shape[0]):
                    if label4[i] == 4 or label4[i] == 9 or label4[i] == 11 or label4[i] == 12 or label4[i] == 14 or \
                            label4[i] == 15 or label4[i] == 19 or label4[i] == 52 or label4[i] == 30 or label4[i] == 31\
                            or label4[i] == 53 or label4[i] == 54 or label4[i] == 80 or label4[i] == 95 or label4[i] == 96\
                            or label4[i] == 72:
                        if unorderd4.get(tindex_u[i]) is None:
                            unorderd4[tindex_u[i]] = 1
                            save_features(feat_all[i + ind1 + ind2 + ind3], label4[i], epoch, 0, 1, 4)
            #  <><><><><><><><><><><><><><><><><><><><><><><><><><><><>  #

            feat_l = feat.copy()
            out0_l = out0.copy()
            out0_u = out0.copy()
            feat_all_l = feat_all.clone()
            if epoch > 5:
                for i in range(6):
                    feat_l[i] = torch.cat((feat[i][0:num], feat[i][num + num_u:num * 2 + num_u]), dim=0)
                    out0_l[i] = torch.cat((out0_l[i][0:num], out0_l[i][num + num_u:num * 2 + num_u]), dim=0)
                    out0_u[i] = torch.cat((out0_u[i][num:num + num_u], out0_u[i][num * 2 + num_u:]), dim=0)
                feat_all_l = torch.cat((feat_all[0:num], feat_all[num + num_u:num * 2 + num_u]), dim=0)
                feat_all_u = torch.cat((feat_all[num:num + num_u], feat_all[num * 2 + num_u:]), dim=0)


            if epoch >= 4:
                feat1 = feat_l[:num * 2].copy()
                feat_all1 = feat_all_l[:num * 2].clone().cpu().detach()
                for i in range(6):
                    feat1[i] = feat1[i].cpu().detach()
                    centre_record(feat1[i][:num], label1, i, dataset_name, flag=True, momentum=0.2)
                    centre_record(feat1[i][num:num * 2], label2, i, dataset_name, flag=False, momentum=0.2)

                centre_record_all(feat_all1, labels, dataset_name, momentum=0.2)

            if epoch > 5:

                if dataset == 'sysu':
                    dpath = 'SYSU-MM01/visible_centre_all.pth'
                elif dataset == 'regdb':
                    dpath = 'RegDB/visible_centre_all.pth'
                cen_all = torch.load(dpath)
                pred = out0_u[0]
                for i in range(5):
                    pred += out0_u[i + 1]
                pred = pred / 6
                pred = torch.softmax(pred, dim=1)
                _, fake_cata_d = torch.max(pred, dim=1)



                ccc = torch.cat((feat_all_u, cen_all.cuda()), 0)
                jacc = jaccard_dis_cal(ccc)
                jac_cendis = jacc[: num_u * 2, num_u * 2:]
                _, fake_cata_jac = torch.min(jac_cendis, dim=1)

                mid = torch.mm(feat_all_u, cen_all.cuda().t())
                mid /= 1.0
                softMask = torch.zeros(mid.shape).cuda()

                nums = torch.zeros(mid.size(0), 1).float().cuda()
                for i in range(mid.size(0)):
                    for j in range(mid.size(0)):
                        if fake_cata_jac[j] == fake_cata_jac[i]:
                            nums[i] += 1
                mid /= nums.clone().expand_as(mid)

                softMask.scatter_(1, fake_cata_jac.cuda().view(-1, 1), 1)
                loss_dce = -(softMask * F.log_softmax(mid, dim=1)).sum(1).mean()
                loss_dsce = -(F.softmax(mid, dim=1) * F.log_softmax(softMask, dim=1)).sum(1).mean()
                loss_den = loss_dce * 0.1 + loss_dsce

            labels = labels.long()
            loss_id = criterion_id(out0_l[0], labels)
            loss_tri_l, batch_acc = criterion_tri(feat_l[0], labels)
            for i in range(args.num_strips - 1):
                loss_id += criterion_id(out0_l[i + 1], labels)
                loss_tri_l += criterion_tri(feat_l[i + 1], labels)[0]

            loss_tri, batch_acc = criterion_tri(feat_all_l[:num * 2], labels)
            loss_tri += loss_tri_l * 2
            loss_id = loss_id
            correct += batch_acc

            if epoch > 5:
                loss = loss_id + loss_tri + loss_den * 2
            else:
                loss = loss_id + loss_tri
        else:
            feat, out0, feat_all = net(input1, input2)
            labels = labels.long()
            loss_id = criterion_id(out0, labels)

            loss_tri, batch_acc = criterion_tri(feat, labels)
            correct += (batch_acc / 2)
            _, predicted = out0.max(1)
            correct += (predicted.eq(labels).sum().item() / 2)
            loss = loss_id + loss_tri * args.w_center

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        train_loss.update(loss.item(), num * 2)
        id_loss.update(loss_id.item(), num * 2)
        tri_loss.update(loss_tri, num * 2)
        total += labels.size(0)


        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0 and epoch < 6:
            loggers('Epoch: [{}][{}/{}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'lr:{:.3f} '
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                    'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                    'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                    'Accu: {:.2f}'.format(epoch, batch_idx, len(trainloader), current_lr,
                                          100. * correct / total, batch_time=batch_time,
                                          train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))
        elif batch_idx % 30 == 0 and epoch >= 6:
            loggers('Epoch: [{}][{}/{}] '
                    'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                    'lr:{:.3f} '
                    'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                    'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                    'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                    'loss_den: {loss_den:.4f} '
                    'Accu: {:.2f}'.format(epoch, batch_idx, len(trainloader), current_lr,
                                          100. * correct / total, batch_time=batch_time,
                                          train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                                          loss_den=loss_den))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)



def pred_label(epoch, cindex_ul, tindex_ul):
    print('==> Precess Unlabeled Data...')
    data_time = AverageMeter()


    net.eval()

    end = time.time()
    num = 2 * 8
    move_index_1 = []
    move_index_2 = []
    if dataset == 'sysu':
        dataset_name = 'SYSU-MM01'
    elif dataset == 'regdb':
        dataset_name = 'RegDB'

    for (batch_idx, (input3, input4, label3, label4)) in enumerate(evalloader_u):

        input3 = Variable(input3.cuda())
        input4 = Variable(input4.cuda())

        data_time.update(time.time() - end)
        cindex = cindex_ul[batch_idx * num:(batch_idx + 1) * num]
        tindex = tindex_ul[batch_idx * num:(batch_idx + 1) * num]

        if args.pcb == 'on':
            _, feat, out0, feat_all = net(input3, input4)

            out1 = out0.copy()
            feat1 = feat.copy()
            cendis_all = [0, 0, 0, 0, 0, 0]

            for i in range(6):
                out1[i] = out1[i].cpu().detach()
                feat1[i] = feat1[i].cpu().detach()


            for i in range(6):
                centre_visible = torch.load(dataset_name + '/visible_centre_' + str(i) + '.pth')
                centre_thermal = torch.load(dataset_name + '/thermal_centre_' + str(i) + '.pth')



                ccc = torch.cat((feat1[i][: num], centre_visible), 0)
                jacc = jaccard_dis_cal(ccc)
                jac_visible = jacc[: feat1[i][: num].size(0), feat1[i][: num].size(0):]
                ccc = torch.cat((feat1[i][num:], centre_thermal), 0)
                jacc = jaccard_dis_cal(ccc)
                jac_thermal = jacc[: feat1[i][num:].size(0), feat1[i][num:].size(0):]

                cendis_all[i] = torch.cat((jac_visible, jac_thermal), dim=0)

            ave_dis = (cendis_all[0] + cendis_all[1] + cendis_all[2] + cendis_all[3] + cendis_all[4] + cendis_all[
                5]) / 6.0

            fake_dis_score, _ = torch.min(ave_dis, dim=1)


            ave = (out1[0] + out1[1] + out1[2] + out1[3] + out1[4] + out1[5]) / 6
            ave = F.softmax(ave, 1)
            fake_label = ave.argmax(dim=1)
            fake_label1 = fake_label[:num]
            fake_label2 = fake_label[num:]



            fake_score, _ = torch.max(ave, dim=1)
            delet_index1 = []
            delet_index2 = []
            for i in range(num * 2):
                if fake_score[i] > 0.8 and ave_dis[i, fake_label[i]] < 0.6:
                    continue
                else:
                    if i < num:
                        delet_index1.append(i)
                    else:
                        delet_index2.append(i - num)

            cindex = np.delete(cindex, delet_index1, 0)
            tindex = np.delete(tindex, delet_index2, 0)

            fake_label1 = fake_label1.cpu().detach().numpy()
            fake_label1 = np.delete(fake_label1, delet_index1, 0)
            fake_label1 = torch.from_numpy(fake_label1).cuda()

            fake_label2 = fake_label2.cpu().detach().numpy()
            fake_label2 = np.delete(fake_label2, delet_index2, 0)
            fake_label2 = torch.from_numpy(fake_label2).cuda()

            update = False

            if min(fake_label1.shape) == 0:
                pass
            else:
                fake_label_record(fake_label1, cindex, True, update, dataset_name, args.trial)
                cindex = cindex.tolist()
                move_index_1 += cindex
            if min(fake_label2.shape) == 0:
                pass
            else:
                fake_label_record(fake_label2, tindex, False, update, dataset_name, args.trial)
                tindex = tindex.tolist()
                move_index_2 += tindex
        if batch_idx % 60 == 0:
            loggers('Epoch:[{}] [{}/{}] '.format(epoch, batch_idx, len(evalloader_u)))

    move_index_1 = list(set(move_index_1))
    move_index_2 = list(set(move_index_2))
    fake_label_move(move_index_1, True, dataset_name, args.trial)
    fake_label_move(move_index_2, False, dataset_name, args.trial)

    del_process(n_color, args.dataset, 1, args.trial)
    del_process(n_thermal, args.dataset, 0, args.trial)
    del_process2(args.dataset, 1, args.trial)
    del_process2(args.dataset, 0, args.trial)


def test(epoch):

    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    if args.pcb == 'on':
        feat_dim = args.num_strips * args.local_feat_dim
    else:
        feat_dim = 2048
    gall_feat = np.zeros((ngall, feat_dim))
    gall_feat_att = np.zeros((ngall, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat, _, _, _ = net(x1=input, x2=input, modal=test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(x1=input, x2=input, modal=test_mode[0])
                gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))


    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0

    query_feat = np.zeros((nquery, feat_dim))
    query_feat_att = np.zeros((nquery, feat_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            if args.pcb == 'on':
                feat, _, _, _ = net(x1=input, x2=input, modal=test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            else:
                feat, feat_att = net(x1=input, x2=input, modal=test_mode[1])
                query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
                query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()

    if args.re_rank == 'random_walk':
        distmat = random_walk(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = random_walk(query_feat_att, gall_feat_att)
    elif args.re_rank == 'k_reciprocal':
        distmat = k_reciprocal(query_feat, gall_feat)
        if args.pcb == 'off': distmat_att = k_reciprocal(query_feat_att, gall_feat_att)
    elif args.re_rank == 'no':

        distmat = -np.matmul(query_feat, np.transpose(gall_feat))
        if args.pcb == 'off': distmat_att = -np.matmul(query_feat_att, np.transpose(gall_feat_att))


    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(distmat, query_label, gall_label)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att = eval_regdb(distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(distmat, query_label, gall_label, query_cam, gall_cam)
        if args.pcb == 'off': cmc_att, mAP_att, mINP_att = eval_sysu(distmat_att, query_label, gall_label, query_cam,
                                                                     gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    if args.pcb == 'off':
        writer.add_scalar('rank1_att', cmc_att[0], epoch)
        writer.add_scalar('mAP_att', mAP_att, epoch)
        writer.add_scalar('mINP_att', mINP_att, epoch)

        return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att
    else:
        return cmc, mAP, mINP



loggers('==> Start Training...')
for epoch in range(start_epoch, 61 - start_epoch):
    if dataset == 'regdb':
        trainset = RegDBData(data_path, args.trial, transform=transform_train)
        trainset_u = RegDBData_u(data_path, args.trial, transform=transform_train)

        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        color_pos_u, thermal_pos_u = GenIdx(trainset_u.train_color_label_u, trainset_u.train_thermal_label_u)
    elif dataset == 'sysu':
        trainset = sysuData(data_path, transform=transform_train)
        trainset_u = sysuData_u(data_path, transform=transform_train)

        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)
        color_pos_u, thermal_pos_u = GenIdx(trainset_u.train_color_label_u, trainset_u.train_thermal_label_u)

    print('==> Preparing Data Loader...')

    if epoch < 6:
        sampler = IdentitySampler(trainset.train_color_label,
                                  trainset.train_thermal_label,
                                  color_pos, thermal_pos, args.num_pos, args.batch_size,
                                  epoch, mode=True)
        sampler_u = IdentitySampler(trainset_u.train_color_label_u,
                                    trainset_u.train_thermal_label_u,
                                    color_pos_u, thermal_pos_u, 4, 4,
                                    epoch, mode=False)
    else:
        sampler = IdentitySampler(trainset.train_color_label,
                                  trainset.train_thermal_label,
                                  color_pos, thermal_pos, 4, 4,
                                  epoch, mode=True)
        sampler_u = IdentitySampler(trainset_u.train_color_label_u,
                                    trainset_u.train_thermal_label_u,
                                    color_pos_u, thermal_pos_u, 4, 4,
                                    epoch, mode=False)

    trainset.cIndex = sampler.index1
    trainset.tIndex = sampler.index2

    trainset_u.cIndex_u = sampler_u.index1
    trainset_u.tIndex_u = sampler_u.index2


    loggers(epoch)

    loader_batch = args.batch_size * args.num_pos
    if epoch < 6:
        trainloader = data.DataLoader(trainset, batch_size=loader_batch,
                                      sampler=sampler, num_workers=args.workers, drop_last=True)
        trainloader_u = data.DataLoader(trainset_u, batch_size=16,
                                        sampler=sampler_u, num_workers=args.workers, drop_last=True)
    else:
        trainloader = data.DataLoader(trainset, batch_size=16,
                                      sampler=sampler, num_workers=args.workers, drop_last=True)
        trainloader_u = data.DataLoader(trainset_u, batch_size=16,
                                        sampler=sampler_u, num_workers=args.workers, drop_last=True)

    centre = 0
    pseudo_labels = 0

    unorderd1 = {}
    unorderd2 = {}
    unorderd3 = {}
    unorderd4 = {}
    unorderd1.clear()
    unorderd2.clear()
    unorderd3.clear()
    unorderd4.clear()


    train(epoch, trainset.cIndex, trainset.tIndex, trainset_u.cIndex_u, trainset_u.tIndex_u, memory_v,
          memory_t)

    if 5 < epoch < 30:
        sampler_e = IdentitySampler(trainset_u.train_color_label_u,
                                    trainset_u.train_thermal_label_u,
                                    color_pos_u, thermal_pos_u, 8, 2,
                                    epoch, mode=False)
        trainset_u.cIndex_u = sampler_e.index1
        trainset_u.tIndex_u = sampler_e.index2
        evalloader_u = data.DataLoader(trainset_u, batch_size=16,
                                       sampler=sampler_e, num_workers=args.workers, drop_last=True)
        pred_label(epoch, sampler_e.index1, sampler_e.index2)

    if epoch > 9 and epoch % 2 == 0:
        loggers('Test Epoch: {}'.format(epoch))


        if args.pcb == 'off':
            cmc, mAP, mINP, cmc_fc, mAP_fc, mINP_fc = test(epoch)
        else:
            cmc_fc, mAP_fc, mINP_fc = test(epoch)

        if cmc_fc[0] > best_acc:
            best_acc = cmc_fc[0]
            best_epoch = epoch
            best_mAP = mAP_fc
            best_mINP = mINP_fc
            state = {
                'net': net.state_dict(),
                'cmc': cmc_fc,
                'mAP': mAP_fc,
                'mINP': mINP_fc,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        if args.pcb == 'off':
            loggers(
                'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

        loggers(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_fc[0], cmc_fc[4], cmc_fc[9], cmc_fc[19], mAP_fc, mINP_fc))
        loggers('Best Epoch [{}], Rank-1: {:.2%} |  mAP: {:.2%}| mINP: {:.2%}'.format(best_epoch, best_acc, best_mAP,
                                                                                      best_mINP))
