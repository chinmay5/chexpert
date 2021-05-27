import torch
from torch import nn
import torch.nn.functional as F


class WeightedBCEWithLogitLoss(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(WeightedBCEWithLogitLoss, self).__init__()
        self.register_buffer('neg_weight', neg_weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        assert input.shape == target.shape, "The loss function received invalid input shapes"
        y_hat = torch.sigmoid(input + 1e-8)
        loss = -1.0 * (self.pos_weight * target * torch.log(y_hat + 1e-6) + self.neg_weight * (1 - target) * torch.log(1 - y_hat + 1e-6))
        # Account for 0 times inf which leads to nan
        loss[torch.isnan(loss)] = 0
        # We average across each of the extra attribute dimensions to generalize it
        loss = loss.mean(dim=1)
        # We use mean reduction for our task
        return loss.mean()


class WeightedBCEWithLogitLossSec(nn.Module):
    def __init__(self, pos_weight, neg_weight):
        super(WeightedBCEWithLogitLossSec, self).__init__()
        self.pos_weight = torch.div(input=neg_weight,other=pos_weight)

    def forward(self, input, target):
        loss = F.binary_cross_entropy_with_logits(input=input, target=target, pos_weight=self.pos_weight, reduction="mean")
        return loss


if __name__ == '__main__':
    y = torch.randint(2, (5, 10)).to(torch.float)
    pos_weight = torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
    neg_weight = torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
    x = torch.randn((5, 10))
    loss_fn = WeightedBCEWithLogitLoss(pos_weight=pos_weight, neg_weight=neg_weight)
    loss1 = loss_fn(x, y)
    loss2 = F.binary_cross_entropy_with_logits(x, y, reduction="mean")
    print(loss1)
    print(loss2)
    print(loss1 - loss2)
    assert torch.allclose(loss1, loss2), "Something wrong with computation"
