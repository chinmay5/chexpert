import os
import time

import math
import torch
from torch.nn import MultiheadAttention
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, GCNConv, GATConv, GATv2Conv
import torch.nn.functional as F

import torch.nn as nn

from environment_setup import PROJECT_ROOT_DIR, GRAPH_FILE_NAME
from models.cnn_base.densenet import DenseNet161, DenseNet121

import numpy as np

from models.our_method.umls_generation.language_embed.embed_cache import EmbedCache
from models.our_method.umls_generation.utils.graph_gen_utils import cuid_map


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


def l2_distance(vec1, vec2):
    return torch.sqrt(torch.sum((vec1 - vec2) ** 2))


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, img_feat, text_feat):
        bs, dim1 = img_feat.shape
        num_nodes, dim2 = text_feat.shape
        assert dim1 == dim2, "Invalid dimension for text or image"
        img_feat = img_feat.unsqueeze(1)
        text_feat = text_feat.unsqueeze(0).repeat(bs, 1, 1)
        # Since the function expects (Seq, BS, Embed_dim), we need to permute. Fixed in newer versions of PyTorch
        img_feat = img_feat.permute(1, 0, 2)  # bs, 14, 128 -> 14, bs, 128
        text_feat = text_feat.permute(1, 0, 2)  # bs, 14, 128 -> 14, bs, 128
        attn_output, attn_wt = self.cross_attention(query=text_feat, key=img_feat, value=img_feat)
        # again performing permutations
        attn_output = attn_output.permute(1, 0, 2)  # 14, bs, 128 -> bs, 14, 128
        return attn_output, attn_wt


class SameModalityAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SameModalityAttention, self).__init__()
        self.cross_attention = MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, feat_dim1, feat_dim2):
        assert feat_dim1.shape == feat_dim2.shape, "Invalid dimension for text or image"
        # Since the function expects (Seq, BS, Embed_dim), we need to permute. Fixed in newer versions of PyTorch
        feat_dim1, feat_dim2 = feat_dim1.unsqueeze(1), feat_dim2.unsqueeze(1)
        attn_output, attn_wt = self.cross_attention(query=feat_dim1, key=feat_dim2, value=feat_dim2)
        # again performing permutations
        attn_output = attn_output.permute(1, 0, 2)  # 14, bs, 128 -> bs, 14, 128
        return attn_output, attn_wt


class CrossModalAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1, dim_feedforward=2048):
        super(CrossModalAttentionLayer, self).__init__()
        self.cross_attn_module = CrossModalAttention(embed_dim=embed_dim, num_heads=num_heads)
        # self.cross_attn_module = SameModalityAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, img_feat, text_feat):
        x_attn, x_attn_wt = self.cross_attn_module(img_feat, text_feat)
        # text_feat = 14, 128 and x_attn = bs, 14, 128
        src = text_feat + self.dropout1(x_attn)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CrossModalTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers=1, dropout=0.4, dim_feedforward=2048):
        super(CrossModalTransformerEncoder, self).__init__()
        self.x_modal_attn_layer = CrossModalAttentionLayer(embed_dim=embed_dim, num_heads=num_heads,
                                                           dim_feedforward=dim_feedforward)
        self_attn_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.self_attn_layers = nn.TransformerEncoder(encoder_layer=self_attn_layer, num_layers=num_layers)

    def forward(self, img_feat, text_feat):
        combined_feat = self.x_modal_attn_layer(img_feat=img_feat, text_feat=text_feat)
        # combined_feat = bs, 14, 128
        # since TransformerEncoder uses CrossModalAttention internally which has (Seq,B, Emd) pattern, hence we need to use
        combined_feat = combined_feat.permute(1, 0, 2)
        combined_feat = self.self_attn_layers(combined_feat)
        # again performing permutations
        combined_feat = combined_feat.permute(1, 0, 2)  # 14, bs, 128 -> bs, 14, 128
        return combined_feat


class GNN_UMLS_Model(nn.Module):

    def __init__(self, in_channels, text_feat_dim, hidden_dim=1024, out_channels=1664, dropout_p=0.2, num_heads=4):
        super(GNN_UMLS_Model, self).__init__()
        self.visual_feature_extractor = DenseNet121(in_channels=in_channels, out_size=out_channels, dropout_p=dropout_p)
        graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'models', 'our_method', 'umls_generation', GRAPH_FILE_NAME))
        print(f"UMLS graph generated using {GRAPH_FILE_NAME}")
        # graph_data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'umls_extraction', 'umls_small_graph.pth'))
        self.edge_index = graph_data.edge_index
        self.node_features = graph_data.x
        self.target_label_indices = torch.as_tensor(graph_data.target_label_indices)

        self.conv1 = GATv2Conv(in_channels=self.node_features.size(1), out_channels=hidden_dim // num_heads,
                               flow='target_to_source', add_self_loops=False, heads=num_heads)
        # self.conv1 = SAGEConv(in_channels=text_feat_dim, out_channels=hidden_dim, flow='target_to_source')

        self.conv2 = GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim, flow='target_to_source',
                               add_self_loops=False)
        # self.conv2 = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim, flow='target_to_source')
        self.combine_concatenated_final = nn.Linear(2 * hidden_dim, hidden_dim)
        self_nodes = []
        for i in range(self.node_features.size(0)):
            self_nodes.append([i, i])
        self.self_nodes = torch.tensor(self_nodes)
        # Now we learn the edge weights based on our Linear Layer network
        # Edge index should be updated at this point else we end up counting the self edges twice
        self.edge_index = torch.cat([self.edge_index, self.self_nodes.T], dim=1).contiguous()
        self.cross_model_attention = CrossModalTransformerEncoder(embed_dim=hidden_dim, num_layers=2, num_heads=4,
                                                                  dropout=0.5)
        self.classification_head = nn.Linear(hidden_dim, 1)
        self.loss_mapping_consistency = 0

    def forward(self, img_feat):
        # Put the graph on the same device as the images
        self.target_label_indices = self.target_label_indices.to(img_feat.device)
        self.node_features = self.node_features.to(img_feat.device)
        self.edge_index = self.edge_index.to(img_feat.device)
        # Extra features to be put on GPU
        # Perform graph convolutions
        # x = self.conv1(x=node_features, edge_index=edge_index, edge_weight=edge_weight)  # W, 200
        # Since one of the edge weights is 0, it would cause issues when performing weight updates
        x = self.conv1(x=self.node_features, edge_index=self.edge_index)  # W, 300
        # res = self.res(self.node_features)
        x = F.leaky_relu(x, negative_slope=0.2)
        layer1_features = x
        # Second layer applied
        x = self.conv2(x=x, edge_index=self.edge_index)  # W, 300
        # x = F.leaky_relu(x)
        # concatenated_features = torch.cat((layer1_features, x), dim=1)
        # x = self.combine_concatenated_final(concatenated_features)
        # Now select only the labels of our interest
        x = torch.index_select(x, dim=0, index=self.target_label_indices)
        x = self.cross_model_attention(img_feat=img_feat, text_feat=x)
        x = self.classification_head(x).squeeze(-1)
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
    model = GNN_UMLS_Model(hidden_dim=128, num_heads=4).cuda()
    out = model(x)
    print(out.shape)
