import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import os

from torch_geometric.nn import GCNConv

from environment_setup import PROJECT_ROOT_DIR
from models.cnn_base.densenet import DenseNet121
from models.graph_base.create_graph import create_graph_data_object


class BaseGNNModel(nn.Module):

    def __init__(self, in_channels, text_feat_dim=300, hidden_dim=1024, out_channels=1664, dropout_p=0.2,
                 load_model_num=None):
        super(BaseGNNModel, self).__init__()

        self.visual_feature_extractor = DenseNet121(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p,
                                                    load_model_num=load_model_num)
        # Create the graph based on label co-occurrences
        create_graph_data_object(debug=False)
        self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'models', 'graph_base', 'base_graph.pth'))
        self.conv1 = GCNConv(in_channels=text_feat_dim, out_channels=hidden_dim, add_self_loops=False)  # Since self loop already present
        self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=out_channels, add_self_loops=False)

    def forward(self, image):
        img_feat = self.visual_feature_extractor(image)
        # Put the graph on the same device as the images
        self.graph_data = self.graph_data.to(img_feat.device)
        edge_index = self.graph_data.edge_index
        node_features = self.graph_data.x
        edge_weight = self.graph_data.edge_attr
        # Perform graph convolutions
        x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        # x = self.conv1(x=node_features, edge_index=edge_index)  # W, 1024
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        # x = self.conv2(x=x, edge_index=edge_index)  # W, 1664
        # img_feat = B, 1664, x = W, 1664
        x = torch.mm(img_feat, x.transpose(0, 1))
        return x

    def get_embedding(self):
        edge_index = self.graph_data.edge_index
        node_features = self.graph_data.x
        edge_weight = self.graph_data.edge_attr
        # Perform graph convolutions
        x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        return x


if __name__ == '__main__':
    x = torch.randn(4, 1, 512, 512)
    y = torch.as_tensor([0, 1, 0, 1])
    model = BaseGNNModel(text_feat_dim=200, in_channels=1)
    out = model(x)
    print(model.get_embedding().shape)
    print(out.shape)
