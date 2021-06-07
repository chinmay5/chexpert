import os
import torch
import torchvision
from torch import nn
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from environment_setup import PROJECT_ROOT_DIR
from models.cnn_base.densenet import DenseNet121


class TFEEncoder(nn.Module):
    def __init__(self, img_feat_dim=512, text_feat_dim=300, num_layers=1):
        super(TFEEncoder, self).__init__()
        transformer_layer = nn.TransformerEncoderLayer(d_model=img_feat_dim + text_feat_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)

    def forward(self, img_feat, word_feat):
        concatenated_feat = torch.cat((img_feat, word_feat), dim=-1)  # B, M, 512+300
        node_feat = self.transformer_encoder(concatenated_feat)
        return node_feat


class GNNNetwork(nn.Module):
    def __init__(self, in_channels, img_feat_dim=512, text_feat_dim=300, hidden_dim=128, num_classes=14, dropout_p=0.2):
        super(GNNNetwork, self).__init__()
        self.visual_feature_extractor = DenseNet121(in_channels=in_channels, out_size=img_feat_dim, dropout_p=dropout_p)
        self.tfe = TFEEncoder(img_feat_dim=img_feat_dim, text_feat_dim=text_feat_dim)
        self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
        self.word_embeds = self.graph_data.x
        self.conv1 = GCNConv(img_feat_dim + text_feat_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.linear_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, image):
        img_feat = self.visual_feature_extractor(image)
        # img_feat = B, 512, word_embeds = M, 300
        img_feat.unsqueeze_(1)  # B, 1, 512
        img_feat = img_feat.repeat(1, self.word_embeds.size(0), 1)  # B, M, 512
        text_feat = self.word_embeds.unsqueeze_(0)  # 1, M, 300
        text_feat = text_feat.repeat(img_feat.size(0), 1, 1)  # B, M, 300
        node_features = self.tfe(img_feat=img_feat, word_feat=text_feat)  # B, M, 512+300
        edge_index = self.graph_data.edge_index
        # Perform graph convolutions
        x = self.conv1(node_features, edge_index)  # B, M, 128
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # B, M, 128
        # This is essentially a mean pooling operation
        x = torch.mean(x, dim=1)  # B, 128
        x = F.relu(x)
        x = self.linear_layer(x)
        return x


if __name__ == '__main__':
    model = GNNNetwork()
    img = torch.randn(4, 3, 224, 224)
    output = model(img)
    print(output.shape)
