import os
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from environment_setup import PROJECT_ROOT_DIR


class WCELossFunc(nn.Module):

    def __init__(self, alpha, beta, num_class):
        super(WCELossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class

    def forward(self, scores, target):
        eps = 0
        pos_count = torch.sum(target).detach()
        total = target.size(0) * target.size(1) + 1
        weight_pos = total / (pos_count + 1)
        weight_neg = total / (total - pos_count)

        loss_list = torch.zeros(len(scores), self.num_class).to(scores.device)

        probs = torch.sigmoid(scores)
        for i in range(len(scores)):
            for j in range(self.num_class):
                loss_list[i][j] = -weight_pos * target[i][j] * torch.pow((1 - probs[i][j]), self.beta) * torch.log(
                    probs[i][j] + eps) \
                                  - weight_neg * (1 - target[i][j]) * torch.pow(probs[i][j], self.beta) * torch.log(
                    1 - probs[i][j] + eps)
        loss = torch.mean(loss_list)
        return loss


def l2_distance(vec1, vec2):
    return sqrt(torch.sum((vec1 - vec2) ** 2).item())


# https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
class WCELossFuncMy(nn.Module):

    def __init__(self, alpha, beta, num_class):
        super(WCELossFuncMy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class
        self.threshold_mat = torch.tensor(np.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mask_table.npy')),
                                          dtype=torch.bool)
        self.labda = 1 / 10

    def forward(self, output, target, logger=None):
        pos_count = torch.sum(target) + 1
        total = target.size(0) * target.size(1) + 1
        weight_pos = total / pos_count  # weight is neg/pos
        weight_neg = total / (total - pos_count + 1)  # weight is neg/pos
        output = torch.sigmoid(output)
        output = output.clamp(min=1e-5, max=1 - 1e-5)

        loss = -weight_pos * (target * torch.log(output)) * torch.pow((1 - output), self.beta) - \
               torch.pow(output, self.beta) * weight_neg * ((1 - target) * torch.log(1 - output))
        # Let us also include the extra loss term just to see it in action
        # pos_locs, neg_loc = torch.nonzero(self.threshold_mat), torch.nonzero(~ self.threshold_mat)
        # pos_sim = sum([l2_distance(vec1=output[:, x], vec2=output[:, y]) for x, y in pos_locs]) / output.size(0)
        # neg_sim = sum([max(0.5 - l2_distance(vec1=output[:, x], vec2=output[:, y]), 0) for x, y in neg_loc]) / output.size(
        #     0)
        # other_loss = pos_sim + neg_sim
        # if logger is not None:
        #     writer, step = logger
        #     writer.add_scalar(tag="euc_pos", scalar_value=pos_sim, global_step=step)
        #     writer.add_scalar(tag="euc_neg", scalar_value=neg_sim, global_step=step)
        #     writer.add_scalar(tag="other_loss", scalar_value=other_loss, global_step=step)
        #     writer.add_scalar(tag="primary_loss", scalar_value=torch.mean(loss).item(), global_step=step)
        return torch.mean(loss)  # + other_loss * self.labda


if __name__ == '__main__':
    alpha = 0.25
    beta = 2
    num_class = 14
    weight_neg = 20
    weight_pos = 25
    loss1 = WCELossFunc(alpha=alpha, beta=beta, num_class=num_class)
    loss2 = WCELossFuncMy(alpha=alpha, beta=beta, num_class=num_class)
    y_labels = torch.randint(high=2, size=(8, 14), dtype=torch.float).cuda()
    logits = torch.randn((8, 14)).cuda()
    l1 = loss1(logits, y_labels)
    l2 = loss2(logits, y_labels)
    print(l1)
    print(l2)
