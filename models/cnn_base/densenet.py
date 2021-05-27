import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, in_channels, out_size, dropout_p):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True, drop_rate=dropout_p)
        self.features = densenet121.features
        self.change_input_channels(in_channels)
        num_ftrs = densenet121.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def change_input_channels(self, in_channels):
        if in_channels == 3:
            return  # No processing required
        self.features.conv0.in_channels = in_channels
        weight = self.features.conv0.weight.detach()
        print(f"Updating pretrained weights for input channel. New size = {in_channels}")
        if in_channels == 1:
            self.features.conv0.weight = nn.parameter.Parameter(weight.sum(1, keepdim=True))
        elif in_channels == 2:
            self.features.conv0.weight = nn.parameter.Parameter(weight[:, :2] * (3.0 / 2.0))
        else:
            raise NotImplementedError(f"No Implementation for in_channels = {in_channels}")


if __name__ == '__main__':
    x = torch.randn((4, 1, 320, 320))
    model = DenseNet121(in_channels=1, out_size=14, dropout_p=0.2)
    out = model(x)
    print(out.shape)