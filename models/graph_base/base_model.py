import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import os

from torch_geometric.nn import GCNConv

from environment_setup import PROJECT_ROOT_DIR
from models.cnn_base.densenet import DenseNet161
from models.graph_base.create_graph import create_graph_data_object
import numpy as np


class BaseGNNModel(nn.Module):

    def __init__(self, in_channels, text_feat_dim=300, hidden_dim=1024, out_channels=1664, dropout_p=0.2,
                 load_model_num=None):
        super(BaseGNNModel, self).__init__()

        self.visual_feature_extractor = DenseNet161(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p,
                                                    load_model_num=load_model_num)
        # Create the graph based on label co-occurrences
        create_graph_data_object(debug=False)
        self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'models', 'graph_base', 'base_graph.pth'))
        self.conv1 = GCNConv(in_channels=text_feat_dim, out_channels=text_feat_dim, add_self_loops=True)  # Since self loop already present
        self.conv2 = GCNConv(in_channels=text_feat_dim, out_channels=hidden_dim, add_self_loops=True)  # Since self loop already present
        self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=out_channels, add_self_loops=True)

    def forward(self, image):
        img_feat = self.visual_feature_extractor(image)
        # Put the graph on the same device as the images
        self.graph_data = self.graph_data.to(img_feat.device)
        edge_index = self.graph_data.edge_index
        node_features = self.graph_data.x
        edge_weight = self.graph_data.edge_attr
        # Perform graph convolutions
        x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 300
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x=x, edge_index=edge_index, edge_weight=edge_weight)
        # img_feat = B, 1664, x = W, 300
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


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNBaseline(nn.Module):

    def __init__(self, in_channels, text_feat_dim=300, hidden_dim=1024, out_channels=1664, dropout_p=0.2,
                 load_model_num=None):
        super(GCNBaseline, self).__init__()

        self.visual_feature_extractor = DenseNet161(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p,
                                                    load_model_num=load_model_num)

        self.gc1 = GraphConvolution(text_feat_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_channels)
        self.relu = nn.LeakyReLU(0.2)

        create_graph_data_object(debug=False)

        A = np.load(os.path.join(PROJECT_ROOT_DIR, 'models', 'graph_base', 'co_occur_directional.npy'))
        self.A = nn.Parameter(torch.from_numpy(A).float())
        self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'models', 'graph_base', 'base_graph.pth'))

    def forward(self, image):
        img_feat = self.visual_feature_extractor(image)
        # Put the graph on the same device as the images
        self.A = self.A.to(img_feat.device)
        self.graph_data = self.graph_data.to(img_feat.device)
        adj = gen_adj(self.A).detach()
        x = self.gc1(self.graph_data.x, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(img_feat, x)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 1, 512, 512)
    y = torch.as_tensor([0, 1, 0, 1])
    model = GCNBaseline(text_feat_dim=200, in_channels=1)
    out = model(x)
    print(out.shape)
