import numpy as np
import random
import copy
import torch


def jaccard_dis_cal(cross_cam_distribute):
    n = cross_cam_distribute.size(0)
    jaccard_dis = torch.zeros((n, n))
    for i in range(n):
        distribute = cross_cam_distribute[i]
        abs_sub = torch.abs(distribute - cross_cam_distribute)
        sum_distribute = distribute + cross_cam_distribute
        intersection = (sum_distribute - abs_sub).sum(dim=1) / 2
        union = (sum_distribute + abs_sub).sum(dim=1) / 2
        jaccard_dis[i, :] = (union - intersection) / union
    return jaccard_dis


def find_neighbor(x, eps):
    N = list()
    temp = x
    for i in range(temp.shape[0]):
        if temp[i] <= eps:
            N.append(i)
    return set(N)


def cal_eps(x, c, label):
    eps = 0.0
    num = x.shape[0]
    for i in range(x.shape[0]):
        temp = torch.sqrt(torch.abs(torch.sum((x[i] - c[label[i]]).pow(2))))
        eps += temp
    eps /= num
    return eps


def centre_calculate(feature, label):
    labels = list(set(label))
    labels.sort(key=label.index)
    centre = torch.zeros(len(labels), feature.size(1)).cuda()  # center初始化
    for i, j in enumerate(labels):
        mid_index = torch.tensor([m for m, n in enumerate(label) if n == j])
        mid_index = mid_index.cuda()
        eee = torch.index_select(feature, 0, mid_index.long())
        cluster_avg = torch.mean(eee, dim=0)
        centre[i] = cluster_avg
    return centre


def DBSCAN(X, eps=0.6, min_Pts=4):
    k = -1
    neighbor_list = []
    omega_list = []
    gama = set([x for x in range(len(X))])
    cluster = [-1 for _ in range(len(X))]
    jaccard_matrix = jaccard_dis_cal(X)
    for i in range(len(X)):
        neighbor_list.append(find_neighbor(jaccard_matrix[i], eps))
        if len(neighbor_list[-1]) >= min_Pts:
            omega_list.append(i)
    omega_list = set(omega_list)
    while len(omega_list) > 0:
        gama_old = copy.deepcopy(gama)
        j = random.choice(list(omega_list))
        k = k + 1
        Q = list()
        Q.append(j)
        gama.remove(j)
        while len(Q) > 0:
            q = Q[0]
            Q.remove(q)
            if len(neighbor_list[q]) >= min_Pts:
                delta = neighbor_list[q] & gama
                deltalist = list(delta)
                for i in range(len(delta)):
                    Q.append(deltalist[i])
                    gama = gama - delta
        Ck = gama_old - gama
        Cklist = list(Ck)
        for i in range(len(Ck)):
            cluster[Cklist[i]] = k
        omega_list = omega_list - Ck
    return cluster


def generate_pseudo_labels(cluster_id, num):
    labels = []
    outliers = 0
    for i, id in enumerate(cluster_id):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num + outliers)
            outliers += 1
    return torch.Tensor(labels).long()
