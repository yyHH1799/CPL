import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd
import numpy as np


def Contrastive_loss(features, label, T=0.5):
    n = label.shape[0]
    similarity_matrix = F.cosine_similarity(features.unsqueeze(1), features.unsqueeze(0), dim=2)

    mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t())).float()


    mask_no_sim = torch.ones_like(mask) - mask


    mask_dui_jiao_0 = torch.ones(n, n) - torch.eye(n, n)


    similarity_matrix = torch.exp(similarity_matrix / T)


    similarity_matrix = similarity_matrix * mask_dui_jiao_0


    sim = mask * similarity_matrix


    no_sim = similarity_matrix - sim


    no_sim_sum = torch.sum(no_sim, dim=1)


    no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
    sim_sum = sim + no_sim_sum_expend
    loss = torch.div(sim, sim_sum)


    loss = mask_no_sim + loss + torch.eye(n, n)


    loss = -torch.log(loss)
    loss = torch.sum(torch.sum(loss, dim=1)) / (2 * n)

    return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda', temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        self.juz = torch.eye(batch_size * 2, batch_size * 2).bool()
        self.register_buffer("negatives_mask", (
            ~self.juz.to(device)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def criterion(out_1, out_2, tau_plus, batch_size, beta, estimator, temperature=0.2):

    out = torch.cat([out_1, out_2], dim=0)
    neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
    old_neg = neg.clone()
    mask = get_negative_mask(batch_size).cuda()
    neg = neg.masked_select(mask).view(2 * batch_size, -1)


    pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
    pos = torch.cat([pos, pos], dim=0)


    if estimator == 'hard':
        N = batch_size * 2 - 2
        imp = (beta * neg.log()).exp()
        reweight_neg = (imp * neg).sum(dim=-1) / imp.mean(dim=-1)
        Ng = (-tau_plus * N * pos + reweight_neg) / (1 - tau_plus)

        Ng = torch.clamp(Ng, min=N * np.e ** (-1 / temperature))
    elif estimator == 'easy':
        Ng = neg.sum(dim=-1)
    else:
        raise Exception('Invalid estimator selected. Please use any of [hard, easy]')


    loss = (- torch.log(pos / (pos + Ng))).mean()

    return loss





T = 0.5
label = torch.tensor([0, 0, 1, 1, 2])
n = label.shape[0]

representations = torch.tensor([[1.0, 2.0, 3.0],
                                [1.2, 2.2, 3.3],
                                [1.3, 2.3, 4.3],
                                [1.5, 2.6, 3.9],
                                [5.1, 2.1, 3.4]])
representations2 = torch.tensor([[10.1, 2.1, 3.1],
                                 [10.3, 2.3, 30.3],
                                 [1.4, 40.4, 1.4],
                                 [1.5, 2.5, 30.5],
                                 [5.5, 50.6, 3.7]])


loss_func = ContrastiveLoss(batch_size=5)
emb_i = torch.rand(5, 512).cuda()
emb_j = torch.rand(5, 512).cuda()



se = 1
