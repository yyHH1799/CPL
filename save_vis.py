import torch
import numpy as np
import math


def init_memory(label, epoch, flag):
    if flag == 1:
        file = 'vis/' + str(epoch) + '/v_' + str(label) + '.pth'
        file2 = 'vis/' + str(epoch) + '/v_' + str(label) + 'ul.pth'
        file3 = 'vis/' + str(epoch) + '/v_' + str(label) + 'r.pth'
    if flag == 0:
        file = 'vis/' + str(epoch) + '/t_' + str(label) + '.pth'
        file2 = 'vis/' + str(epoch) + '/t_' + str(label) + 'ul.pth'
        file3 = 'vis/' + str(epoch) + '/t_' + str(label) + 'r.pth'
    zeros = torch.zeros([1, 1536])
    torch.save(zeros, file)
    torch.save(zeros, file2)
    torch.save(zeros, file3)


def save_features(feature, label, epoch, flag, flag2, flag3):
    label = label.numpy()
    if flag == 1:
        if flag2 == 0:
            if flag3 == 0:
                file = 'vis/' + str(epoch) + '/v_' + str(label) + '.pth'
            elif flag3 == 1:
                file = 'vis/' + str(epoch) + '/v_' + str(label) + 'r.pth'
        elif flag2 == 1:
            file = 'vis/' + str(epoch) + '/v_' + str(label) + 'ul.pth'
    if flag == 0:
        if flag2 == 0:
            if flag3 == 0:
                file = 'vis/' + str(epoch) + '/t_' + str(label) + '.pth'
            elif flag3 == 1:
                file = 'vis/' + str(epoch) + '/t_' + str(label) + 'r.pth'
        elif flag2 == 1:
            file = 'vis/' + str(epoch) + '/t_' + str(label) + 'ul.pth'
    now_save = torch.load(file)
    feat = feature.cpu()
    tosave = torch.cat((now_save, feat.view(1, 1536)), 0)
    torch.save(tosave, file)
    return 0


