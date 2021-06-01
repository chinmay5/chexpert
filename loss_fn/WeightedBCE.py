import torch
import torch.nn as nn


class WCELossFunc(nn.Module):

    def __init__(self, alpha, beta, num_class):
        super(WCELossFunc, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class

    def forward(self, scores, target):

        pos_count = torch.sum(target).detach()
        total = target.size(0) * target.size(1)
        weight_pos = total / pos_count
        weight_neg = total / (total - pos_count)

        loss_list = torch.zeros(len(scores), self.num_class).to(scores.device)

        probs = torch.sigmoid(scores)
        for i in range(len(scores)):
            for j in range(self.num_class):
                loss_list[i][j] = -weight_pos * target[i][j] * torch.pow((1 - probs[i][j]), self.beta) * torch.log(
                    probs[i][j]) \
                                  - weight_neg * (1 - target[i][j]) * torch.pow(probs[i][j],
                                                                                self.beta) * torch.log(
                    1 - probs[i][j])

        loss = torch.mean(loss_list)
        return loss


class WCELossFuncMy(nn.Module):

    def __init__(self, alpha, beta, num_class):
        super(WCELossFuncMy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_class = num_class

    def forward(self, scores, target):
        pos_count = torch.sum(target)
        total = target.size(0) * target.size(1)
        weight_pos = total / pos_count
        weight_neg = total / (total - pos_count)
        probs = torch.sigmoid(scores)
        loss_tensor = -weight_pos * target * torch.pow((1 - probs), self.beta) * torch.log(probs) \
                      - weight_neg * (1 - target) * torch.pow(probs, self.beta) * torch.log(1 - probs)

        loss = torch.mean(loss_tensor)
        return loss


if __name__ == '__main__':
    alpha = 0.1
    beta = 0.7
    num_class = 14
    weight_neg = 20
    weight_pos = 25
    loss1 = WCELossFunc(alpha=alpha, beta=beta, num_class=num_class)
    loss2 = WCELossFuncMy(alpha=alpha, beta=beta, num_class=num_class)
    y_labels = torch.randint(high=2, size=(8, 14)).cuda()
    logits = torch.randn((8, 14)).cuda()
    l1 = loss1(logits, y_labels)
    l2 = loss2(logits, y_labels)
    assert torch.isclose(l1, l2), "Move to default"
