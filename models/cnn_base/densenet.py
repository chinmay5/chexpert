import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from environment_setup import PROJECT_ROOT_DIR


class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self, in_channels, out_size, dropout_p, load_model_num=None):
        super(DenseNet121, self).__init__()
        densenet121 = torchvision.models.densenet121(pretrained=True, drop_rate=dropout_p)
        self.features = densenet121.features
        self.change_input_channels(in_channels)
        num_ftrs = densenet121.classifier.in_features
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size)
        )
        self.load_features(load_model_num=load_model_num)

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

    def load_features(self, load_model_num):
        if load_model_num is None:
            return
        # Else load everything now
        print(f"Loading pretrained model:  model{load_model_num}.pth")
        checkpoint_name = os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", f"model{load_model_num}.pth")
        assert os.path.exists(checkpoint_name), "Please make sure pretrained model is present for loading"
        modelCheckpoint = torch.load(checkpoint_name)
        pretrained_dict = modelCheckpoint['state_dict']
        model_dict = self.features.state_dict()
        # Let us try the filtering thing here once
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the model now
        self.features.load_state_dict(model_dict)
        # Let us freeeze the weights as well for debugging
        # TODO: Check if this is needed
        # for param in self.features.parameters():
        #     param.requires_grad = False

    def get_embedding(self):
        return self.classifier[0].weight



if __name__ == '__main__':
    x = torch.randn((4, 1, 320, 320))
    model = DenseNet121(in_channels=1, out_size=14, dropout_p=0.2, load_model_num=None)
    out = model(x)
    print(out.shape)
    print(model.classifier[0].weight.shape)
    def check_model_size(model):
        num_params = 0
        traininable_param = 0
        for param in model.parameters():
            num_params += param.numel()
            if param.requires_grad:
                traininable_param += param.numel()
        print("[Network  Total number of parameters : %.3f M" % (num_params / 1e6))
        print(
            "[Network  Total number of trainable parameters : %.3f M"
            % (traininable_param / 1e6)
        )
    check_model_size(model=model)
