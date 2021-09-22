import os
import time

import math
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, GCNConv, GATConv
import torch.nn.functional as F

import torch.nn as nn

from environment_setup import PROJECT_ROOT_DIR
from models.cnn_base.densenet import DenseNet161

import numpy as np

from models.language_processing.embed_cache import EmbedCache
from models.our_method.graph_gen_utils import cuid_map


def generate_aux_graph():
    adj_matrix = np.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_data_prob.npy'))
    # Get node features first
    cache = EmbedCache(strategy="bio")
    node_names = list(cuid_map.keys())
    # The matrix was created in the same sequence as the cuid_map.keys()
    node_embeddings = cache.get_embedding(node_names=node_names)
    x = torch.as_tensor(node_embeddings, dtype=torch.float)
    edge_index = torch.as_tensor(
        [[int(e[0]), int(e[1])] for e in zip(*adj_matrix.nonzero())],
        dtype=torch.long)
    edge_features = [[adj_matrix[int(e[0])][int(e[1])]] for e in zip(*adj_matrix.nonzero())]
    edge_features = torch.as_tensor(np.concatenate(edge_features), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    torch.save(data, os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_prob_mat.pth'))


def l2_distance(vec1, vec2):
    return torch.sqrt(torch.sum((vec1 - vec2) ** 2))

class GNN_UMLS_Model(nn.Module):

    def __init__(self, in_channels, text_feat_dim, hidden_dim=1024, out_channels=1664, dropout_p=0.2,
                 load_model_num=None):
        super(GNN_UMLS_Model, self).__init__()

        self.visual_feature_extractor = DenseNet161(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p,
                                                    load_model_num=load_model_num)
        # self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'new_graph.pth'))
        graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'new_graph_big.pth'))
        self.edge_index = graph_data.edge_index
        # TODO: Check if allowing a feature update of the nodes is helpful or not
        self.node_features = graph_data.x.requires_grad_(False)
        self.node_mask = graph_data.mask
        self.conv1 = GATConv(in_channels=text_feat_dim, out_channels=hidden_dim, aggr='add', flow='target_to_source')
        self.conv2 = GATConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='add', flow='target_to_source')
        self.conv3 = GATConv(in_channels=hidden_dim, out_channels=out_channels, aggr='add', flow='target_to_source')

        self_nodes = []
        for i in range(self.node_features.size(0)):
            self_nodes.append([i, i])
        self.self_nodes = torch.tensor(self_nodes)
        # Now we learn the edge weights based on our Linear Layer network
        # Edge index should be updated at this point else we end up counting the self edges twice
        self.edge_index = torch.cat([self.edge_index, self.self_nodes.T], dim=1).contiguous()
        # self.conv1 = GraphConv(in_channels=text_feat_dim, out_channels=hidden_dim, aggr='mean', add_self_loops=True)
        # self.conv2 = GraphConv(in_channels=hidden_dim, out_channels=out_channels, aggr='mean', add_self_loops=True)  # Since self loop already present

    def forward(self, image):
        img_feat = self.visual_feature_extractor(image)
        # Put the graph on the same device as the images
        self.node_mask = self.node_mask.to(image.device)
        self.node_features = self.node_features.to(image.device)
        self.edge_index = self.edge_index.to(image.device)
        # Extra features to be put on GPU

        # Perform graph convolutions
        # x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 200
        # Since one of the edge weights is 0, it would cause issues when performing weight updates
        x = self.conv1(x=self.node_features, edge_index=self.edge_index)  # W, 300
        x = F.leaky_relu(x, negative_slope=0.2)
        # x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
        x = self.conv2(x=x, edge_index=self.edge_index)  # W, 1024
        # # Now select only the labels of our interest
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv3(x=x, edge_index=self.edge_index)  # W, 1024
        x = x[self.node_mask]

        # img_feat = B, 1664, x = 14, 1664
        x = torch.mm(img_feat, x.transpose(0, 1))
        return x


# class GNN_UMLS_Model(nn.Module):
#
#     def __init__(self, in_channels, text_feat_dim, hidden_dim=1024, out_channels=1664, dropout_p=0.2,
#                  load_model_num=None):
#         super(GNN_UMLS_Model, self).__init__()
#
#         self.visual_feature_extractor = DenseNet161(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p,
#                                                     load_model_num=load_model_num)
#         # self.graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'new_graph.pth'))
#         graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'new_graph_big.pth'))
#         self.edge_index = graph_data.edge_index
#         # TODO: Check if allowing a feature update of the nodes is helpful or not
#         self.node_features = graph_data.x.requires_grad_(False)
#         self.node_mask = graph_data.mask
#         self.warmup_node_embeds()
#         self.conv1 = GCNConv(in_channels=text_feat_dim, out_channels=hidden_dim, aggr='add', flow='target_to_source')
#         self.conv2 = GCNConv(in_channels=hidden_dim, out_channels=hidden_dim, aggr='add', flow='target_to_source')
#         self.conv3 = GCNConv(in_channels=hidden_dim, out_channels=out_channels, aggr='add', flow='target_to_source')
#         self.edge_wt_computation = nn.Sequential(
#             nn.Linear(in_features=2 * text_feat_dim, out_features=text_feat_dim),
#             nn.ReLU(),
#             nn.Linear(in_features=text_feat_dim, out_features=1),
#             nn.Sigmoid()
#         )
#         self_nodes = []
#         self_node_weights = []
#         for i in range(self.node_features.size(0)):
#             self_nodes.append([i, i])
#             self_node_weights.append(torch.tensor([1], dtype=torch.float))
#         self.self_nodes = torch.tensor(self_nodes)
#         self.self_node_wt = torch.cat(self_node_weights)
#         # Now we learn the edge weights based on our Linear Layer network
#         edge_attr = []
#         for u,v in self.edge_index.T:
#             edge_attr.append(torch.cat([self.node_features[u], self.node_features[v]], dim=0))
#             # edge_attr.append(torch.dot(self.node_features[u], self.node_features[v]) / (torch.norm(self.node_features[u]) * torch.norm(self.node_features[v])))
#         self.edge_attr = torch.stack(edge_attr)
#         # Edge index should be updated at this point else we end up counting the self edges twice
#         self.edge_index = torch.cat([self.edge_index, self.self_nodes.T], dim=1).contiguous()
#         # self.conv1 = GraphConv(in_channels=text_feat_dim, out_channels=hidden_dim, aggr='mean', add_self_loops=True)
#         # self.conv2 = GraphConv(in_channels=hidden_dim, out_channels=out_channels, aggr='mean', add_self_loops=True)  # Since self loop already present
#
#     def forward(self, image):
#         img_feat = self.visual_feature_extractor(image)
#         img_feat = F.normalize(img_feat)
#         # Put the graph on the same device as the images
#         self.node_mask = self.node_mask.to(image.device)
#         self.node_features = self.node_features.to(image.device)
#         self.edge_index = self.edge_index.to(image.device)
#         # Extra features to be put on GPU
#         self.edge_attr = self.edge_attr.to(device=image.device)
#         self.self_node_wt = self.self_node_wt.to(device=image.device)
#         # We also add the self loops here
#
#         edge_weights = self.edge_wt_computation(self.edge_attr).squeeze()
#         # edge_weights = self.edge_attr
#         self.edge_weights = torch.cat([edge_weights, self.self_node_wt], dim=0).to(device=image.device)
#         # Threshold the edge values to stop negative transfer
#         self.edge_weights[self.edge_weights < 0.4] = 0
#         # Perform graph convolutions
#         # x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 200
#         # Since one of the edge weights is 0, it would cause issues when performing weight updates
#         x = self.conv1(x=self.node_features, edge_index=self.edge_index, edge_weight=self.edge_weights)  # W, 300
#         # x = self.bn1(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         # x = self.conv2(x=x, edge_index=edge_index, edge_weight=edge_weight)  # W, 1024
#         x = self.conv2(x=x, edge_index=self.edge_index, edge_weight=self.edge_weights)  # W, 1024
#         # # Now select only the labels of our interest
#         # x = self.bn2(x)
#         x = F.leaky_relu(x, negative_slope=0.2)
#         x = self.conv3(x=x, edge_index=self.edge_index, edge_weight=self.edge_weights)  # W, 1024
#         x = x[self.node_mask]
#         x = F.normalize(x)
#
#         # img_feat = B, 1664, x = 14, 1664
#         x = torch.mm(img_feat, x.transpose(0, 1))
#         return x
#
#     def warmup_node_embeds(self):
#         print("Warming up the node embeddings")
#         transit_layer = nn.Sequential(
#             nn.Linear(200, 200),
#             nn.ReLU(),
#             nn.Linear(200, 200)
#         )
#         warmup_optim = torch.optim.Adam(lr=1e-3, params=transit_layer.parameters())
#         threshold_mat = torch.tensor(np.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mask_table.npy')),
#                                      dtype=torch.bool)
#         # Let us also include the extra loss term just to see it in action
#         pos_locs, neg_loc = torch.nonzero(threshold_mat), torch.nonzero(~ threshold_mat)
#         node_features = self.node_features.clone()
#         for _ in range(100):
#             node_features = transit_layer(node_features)
#             select_nodes = node_features[self.node_mask]
#             pos_sim = torch.mean(torch.tensor([l2_distance(vec1=select_nodes[:, x], vec2=select_nodes[:, y]) for x, y in pos_locs]))
#             neg_sim = torch.mean(torch.tensor([max(0.5 - l2_distance(vec1=select_nodes[:, x], vec2=select_nodes[:, y]), 0) for x, y in neg_loc]))
#             loss = pos_sim + neg_sim
#             loss.requires_grad = True
#             loss.backward(retain_graph=True)
#             warmup_optim.step()
#         self.node_features = transit_layer(self.node_features).detach()



if __name__ == '__main__':
    # generate_aux_graph()
    x = torch.randn(2, 1, 512, 512).cuda()
    model = GNN_UMLS_Model(text_feat_dim=200, in_channels=1).cuda()
    out = model(x)
    print(out.shape)
