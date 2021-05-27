import re

import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data

import networkx as nx

from dataset.ChexpertDataloader import label_columns
from environment_setup import PROJECT_ROOT_DIR, read_config

import plotly.graph_objects as go
import plotly.io as pio

from models.language_processing.embed_cache import EmbedCache


def plot_cooccurence():
    # plot co-occurance matrix
    co_occurrence_mat, co_occur_directional = get_cooccurrances()
    plt.figure(figsize=(20, 5), dpi=100)
    plt.subplot(131)
    plt.yticks(np.arange(len(label_columns)), label_columns)
    plt.xticks(np.arange(len(label_columns)), label_columns, rotation='vertical')
    plt.ylim(-0.5, len(label_columns) - 0.5)
    plt.gca().invert_yaxis()
    plt.imshow(co_occurrence_mat)
    plt.subplot(132)
    plt.xticks(np.arange(len(label_columns)), label_columns, rotation='vertical')
    plt.ylim(-0.5, len(label_columns) - 0.5)
    plt.gca().invert_yaxis()
    plt.imshow(co_occur_directional)
    plt.colorbar()
    plt.savefig(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur.png"))
    plt.show()


def get_nx_graph():
    data = create_graph_data_object()
    # We do not use directed graphs here since inverse relations would not exist and we would be adding it explicitly.
    G = nx.Graph()

    G.add_nodes_from(list(range(len(label_columns))))  # We just want to use integer values here

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
    return G


def plot_graph():
    m_graph = get_nx_graph()
    pos = nx.spring_layout(m_graph, k=0.3)
    nx.draw(m_graph, pos=pos, with_labels=False)
    labels = {idx: val for idx, val in enumerate(label_columns)}
    nx.draw_networkx_labels(m_graph, pos, labels, font_size=8, font_color='r')
    plt.savefig(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "graph_structure.png"), dpi=1000)


def get_cooccurrances():
    co_occur_mat = np.load(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur_mat.npy"))
    co_occur_directional = np.load(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur_directional.npy"))
    return co_occur_mat, co_occur_directional


def create_graph_data_object(debug=True):
    cache = EmbedCache(strategy="bio")
    co_occur_directional = np.load(os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "co_occur_directional.npy"))
    # From the adjacency matrix, we would need to convert it into a COO format.
    edge_index = torch.as_tensor(
        [[int(e[0]), int(e[1])] for e in zip(*co_occur_directional.nonzero())],
        dtype=torch.long)
    edge_features = [[co_occur_directional[int(e[0])][int(e[1])]] for e in zip(*co_occur_directional.nonzero())]
    edge_features = torch.as_tensor(np.concatenate(edge_features), dtype=torch.float).unsqueeze(1)
    # We would also need word features
    new_node_names = []
    for x in label_columns:
        new_node_names.append(re.sub(r'[0-9\(\)\/:;,-]', ' ', x))
    if debug:
        x = torch.as_tensor(range(len(label_columns)), dtype=torch.float)
    else:
        node_embeddings = cache.get_embedding(node_names=new_node_names)
        x = torch.as_tensor(node_embeddings, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.t().contiguous(), edge_attr=edge_features)
    torch.save(data, os.path.join(PROJECT_ROOT_DIR, 'models', 'graph_base', 'base_graph.pth'))
    return data


def plot_3d_graph(uncertainty_labels):
    # https://deepnote.com/@deepnote/3D-network-visualisations-using-plotly-oYxeN6UXSye_3h_ulKV2Dw
    m_graph = get_nx_graph()
    Num_nodes = len(m_graph.nodes())
    edge_list = m_graph.edges()

    pio.renderers.default = "browser"

    # set the argument 'with labels' to False so you have unlabeled graph
    pos = nx.spring_layout(m_graph, dim=3, seed=42)
    # we need to seperate the X,Y,Z coordinates for Plotly
    x_nodes = [pos[i][0] for i in range(Num_nodes)]  # x-coordinates of nodes
    y_nodes = [pos[i][1] for i in range(Num_nodes)]  # y-coordinates
    z_nodes = [pos[i][2] for i in range(Num_nodes)]  # z-coordinates
    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # need to fill these with all of the coordiates
    for edge in edge_list:
        # format: [beginning,ending,None]
        x_coords = [pos[edge[0]][0], pos[edge[1]][0], None]
        x_edges += x_coords

        y_coords = [pos[edge[0]][1], pos[edge[1]][1], None]
        y_edges += y_coords

        z_coords = [pos[edge[0]][2], pos[edge[1]][2], None]
        z_edges += z_coords
    # create a trace for the edges
    trace_edges = go.Scatter3d(x=x_edges,
                               y=y_edges,
                               z=z_edges,
                               mode='lines',
                               line=dict(color='black', width=2),
                               hoverinfo='none')
    # create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers',
                               marker=dict(symbol='circle',
                                           size=10,
                                           color='green',
                                           line=dict(color='black', width=0.5)
                                           ),
                               text=label_columns,
                               hoverinfo='text')
    # we need to set the axis for the plot
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout(title=f"Label Co-occurrence Graph for {uncertainty_labels}",
                       showlegend=False,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis),
                                  ),
                       margin=dict(t=100),
                       hovermode='closest')
    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, os.path.join(PROJECT_ROOT_DIR, "models", "graph_base", "baseline_co_occurrence.html"))


if __name__ == '__main__':
    parser = read_config()
    uncertainty_labels = parser['data'].get('uncertainty_labels')
    plot_3d_graph(uncertainty_labels=uncertainty_labels)
    # plot_cooccurence()