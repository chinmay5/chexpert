import pickle
import sys
from collections import defaultdict

import csv

import pandas as pd
import requests
import torch
import numpy as np
import os
import networkx as nx

import plotly.graph_objects as go
import plotly.io as pio
from matplotlib import pyplot as plt
from networkx import shortest_path
from networkx.exception import NetworkXNoPath
from tabulate import tabulate
from torch_geometric.data import Data
from tqdm import tqdm

pd.set_option("display.max_rows", None, "display.max_columns", None)

# This import is needed for pickling
from models.our_method.graph_gen_utils import CuidInfo

from torch_geometric.utils import degree, subgraph, contains_self_loops, contains_isolated_nodes, is_undirected

from environment_setup import PROJECT_ROOT_DIR

from models.our_method.graph_gen_utils import cuid_map

device = torch.device("cpu")

label_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper.pkl'), 'rb'))
cuid_2_common_name = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "rb"))
con2id = {v: k for k, v in label_map.items()}
cuid_chexpert_target_labels = [con2id[cuid] for cuid in cuid_map.values()]


# Has node_id to cuid mapping

def prune_leaf_nodes(data):
    # Obsolete code but works. Remember, when it works, please don't touch!!!
    deg = degree(data.edge_index[1], data.num_nodes)
    mask = deg >= 2
    edge_index, edge_attr = subgraph(mask, data.edge_index, data.edge_attr,
                                     relabel_nodes=False, num_nodes=data.num_nodes)
    data.x = data.x[mask]
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    print("leaves pruned")
    return data


def save_cuid_cuid_rel():
    data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
    data = data.cuda()
    deg = degree(data.edge_index[1], data.num_nodes)
    mask = deg >= 4
    edge_index, edge_attr = subgraph(mask, data.edge_index, data.edge_attr,
                                     relabel_nodes=False, num_nodes=data.num_nodes)
    nodes = torch.nonzero(mask, as_tuple=False)[:, 0]
    data.edge_index = edge_index
    # We need to load two sets of files.
    # The first one is holding cuid based relationships. Then, we can query it and get all possible relationships
    # for the label map
    the_huge_connectivity_dict = pickle.load(
        open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'huge_cuid_cuid_obj.pkl'), 'rb'))
    csv_list = []
    # second csv file is the label_map that we can use to get cuids
    # Since the graph is undirected, we are fine
    # assert is_undirected(edge_index), "The graph should be undirected for the proposed logic to work here"
    for src, tgt in tqdm(edge_index.T):
        src, tgt = label_map[src.item()], label_map[tgt.item()]
        candidate_cuid_objects = the_huge_connectivity_dict[src]
        for cuid_obj in candidate_cuid_objects:
            if cuid_obj.cuid == tgt:
                csv_list.append([src, cuid_obj.rel, cuid_obj.rela, tgt])
                break

    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', "relationships.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["src_cuid", "rel", "rela", "target_cuid"])
        writer.writerows(csv_list)


def get_nx_graph(load_edge_weights=False, prune_nodes=False, data=None):
    if data is None:
        data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
        # We replace the node embeddings with just the node sequences. This works since we ensure sequential numbering
        # for the nodes all through.
        data.x = torch.as_tensor(list(range(data.x.size(0))), dtype=torch.int)
    if prune_nodes:
        data = prune_leaf_nodes(data=data)
    # We do not use directed graphs here since inverse relations would not exist and we would be adding it explicitly.
    G = nx.Graph()

    G.add_nodes_from(data.x.squeeze().tolist())  # Since these are basically cuid integer maps

    values = {}
    for key, item in data:
        if torch.is_tensor(item):
            values[key] = item.squeeze().tolist()
        else:
            values[key] = item
        if isinstance(values[key], (list, tuple)) and len(values[key]) == 1:
            values[key] = item[0]

    if load_edge_weights:
        for (u, v), wt in zip(data.edge_index.t().tolist(), data.edge_attr.tolist()):
            G.add_edge(u, v, foo=wt)
    else:
        for i, (u, v) in enumerate(data.edge_index.t().tolist()):
            G.add_edge(u, v)
    return G, data


def map_cuid_2_names(important_nodes):
    # select * from MRCONSO where
    # TS='P' and
    # STT='PF' and
    # ISPREF='Y' and
    # LAT='ENG'
    # labels = defaultdict(str)
    # with open(os.path.join(PROJECT_ROOT_DIR, 'data', 'umls', 'META', 'MRCONSO.RRF')) as file:
    #     # Pass through outer loop once and check values in the inner
    #     reader = csv.reader(file, delimiter='|')
    #     for row in tqdm(reader, desc="reading"):
    #         for node, cuid in important_nodes.items():
    #             if all([row[1] == 'ENG', row[6] == 'Y', row[4] == 'PF', row[2] == 'P', row[0] == cuid]):
    #                 labels[node] = row[-5]
    # return labels
    labels = defaultdict(str)
    for node, cuid in tqdm(important_nodes.items()):
        labels[node] = cuid_2_common_name[cuid]
    return labels


def plot_3d_graph(chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, edge_text=None, debug=False,
                  out_file='sample.html'):
    # Keep track of `chexpert label` nodes since we would like to have them distinct

    if debug:
        labels = node_2_cuid_map
    else:
        labels = map_cuid_2_names(node_2_cuid_map)

    node_names = [y for _, y in sorted(labels.items())]  # nodes are sorted to order from (0,1,2...) to map edge indices

    # We try ro use plotly to visualize the graph now

    pio.renderers.default = "browser"

    # set the argument 'with labels' to False so you have unlabeled graph
    pos = nx.spring_layout(m_graph, dim=3, seed=42)
    # we need to seperate the X,Y,Z coordinates for Plotly
    # x_nodes = [pos[i][0] for i in range(Num_nodes)]  # x-coordinates of nodes
    # y_nodes = [pos[i][1] for i in range(Num_nodes)]  # y-coordinates
    # z_nodes = [pos[i][2] for i in range(Num_nodes)]  # z-coordinates

    # Since some nodes are prunned, we can not use `range(num_nodes)` here
    x_nodes = [pos[i][0] for i in pos.keys()]  # x-coordinates of nodes
    y_nodes = [pos[i][1] for i in pos.keys()]  # y-coordinates
    z_nodes = [pos[i][2] for i in pos.keys()]  # z-coordinates

    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges = []
    y_edges = []
    z_edges = []

    # need to fill these with all of the coordiates
    for edge in tqdm(edge_list):
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
                               text=edge_text,
                               hoverinfo='none')
    # create a trace for the nodes
    trace_nodes = go.Scatter3d(x=x_nodes,
                               y=y_nodes,
                               z=z_nodes,
                               mode='markers',
                               text=node_names,
                               hoverinfo='text')
    # we need to set the axis for the plot
    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')
    # also need to create the layout for our plot
    layout = go.Layout(title="UMLS graph representation",
                       showlegend=False,
                       scene=dict(xaxis=dict(axis),
                                  yaxis=dict(axis),
                                  zaxis=dict(axis),
                                  ),
                       margin=dict(t=100),
                       hovermode='closest')
    print("Layout created")
    # Include the traces we want to plot and create a figure
    data = [trace_edges, trace_nodes]
    # To make our central nodes more visible, we can create another trace for the same.
    for node_num, idx in chexpert_labels_map.items():
        text = labels[node_num]
        data.append(
            go.Scatter3d(x=[x_nodes[idx]],
                         y=[y_nodes[idx]],
                         z=[z_nodes[idx]],
                         mode='markers',
                         marker=dict(symbol='circle',
                                     size=10,
                                     color='green',
                                     line=dict(color='black', width=0.5)
                                     ),
                         text=text,
                         hoverinfo='text')
        )
    fig = go.Figure(data=data, layout=layout)
    pio.write_html(fig, os.path.join(PROJECT_ROOT_DIR, "dataset", out_file))


def bootstrap_data(load_edge_weights=False, prune_nodes=False, data=None):
    # https://deepnote.com/@deepnote/3D-network-visualisations-using-plotly-oYxeN6UXSye_3h_ulKV2Dw
    m_graph, data = get_nx_graph(load_edge_weights=load_edge_weights, prune_nodes=prune_nodes, data=data)
    # Num_nodes = len(m_graph.nodes())
    edge_list = m_graph.edges()
    node_2_cuid_map = {node: label_map[node] for node in m_graph.nodes()}
    chexpert_label_nodes = [node_num for node_num, cuid in node_2_cuid_map.items() for node_cuids in cuid_map.values()
                            if cuid in node_cuids]
    chexpert_labels_map = {x: np.where(data.x.numpy() == x)[0].item() for x in chexpert_label_nodes}
    return chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, data


def create_giant_dict():
    # select * from MRCONSO where
    # TS='P' and
    # STT='PF' and
    # ISPREF='Y' and
    # LAT='ENG'
    labels = defaultdict(str)
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRCONSO.RRF')) as file:
        # Pass through outer loop once and check values in the inner
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader, desc="reading"):
            if all([row[1] == 'ENG', row[6] == 'Y', row[4] == 'PF', row[2] == 'P']):
                labels[row[0]] = row[-5]
    pickle.dump(labels, open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "wb"))
    return labels


def _tabulate_and_print(path_len, cuid_list):
    cuids = [label_map[x] for x in cuid_list]
    labels_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "rb"))
    headers = [labels_dict[x] for x in cuids]
    table = tabulate(path_len, headers, tablefmt="fancy_grid")
    print(table)
#     Let us also create a similarity mask table
    mask_table = path_len <= 3
    np.save(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mask_table'), mask_table)
    return mask_table


def cuid_cuid_shortest_path(networkx_graph, cuid_list, store_path_prob_matrix=False):
    path_len = torch.zeros((len(cuid_list), len(cuid_list)))
    avg_shortst_path = []
    for i, cuid in enumerate(cuid_list):
        # for j in range(i + 1, len(cuid_list)):
        for j in range(i, len(cuid_list)):
            src_node, target_node = cuid, cuid_list[j]
            if i == j:
                path_len[i, j] = 0
                continue
            try:
                path_len[i, j] = path_len[j, i] = len(
                    shortest_path(networkx_graph, source=src_node,
                                  target=target_node)) - 2  # Since it includes source and target nodes
            except NetworkXNoPath:
                path_len[i, j] = path_len[j, i] = float("inf")
            avg_shortst_path.append(path_len[i, j])
    print(f"Avg path length is {sum(avg_shortst_path) / len(avg_shortst_path)}")
    _tabulate_and_print(path_len=path_len, cuid_list=cuid_list)
    path_len.fill_diagonal_(1e-20)
    prob_matrix = torch.softmax(input=path_len, dim=1)
    # Higher softmax => larger distance. So, take an inverse of that
    prob_matrix = 1 - prob_matrix
    prob_matrix.fill_diagonal_(1.0)
    if store_path_prob_matrix:
        print("storing the path length matrix")
        np.save(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_data_prob.npy'), prob_matrix)
    # print(path_len)


def _relabel_edge_indices(edge_index, edge_attr, selected_nodes):
    for idx, val in enumerate(selected_nodes):
        indices = torch.where(edge_index[0] == val)
        edge_index[0][indices] = idx
        # Now for the destination nodes
        indices = torch.where(edge_index[1] == val)
        edge_index[1][indices] = idx
    return edge_index, edge_attr


def trim_nodes_based_on_shortest_path(networkx_graph, cuid_list, prune_nodes, save_new_dataobject,
                                      include_incident_en_route=False, save_filename='new_graph.pth',
                                      save_csv_file=False):
    assert not prune_nodes, "The method works only when input nodes are not prunned!!!"
    data = torch.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
    mask = torch.zeros((data.x.size(0)), dtype=torch.bool)
    for i, cuid in enumerate(cuid_list):
        # for j in range(i + 1, len(cuid_list)):
        for j in range(len(cuid_list)):
            src_node, target_node = cuid, cuid_list[j]
            try:
                nodes_enroute = shortest_path(networkx_graph, source=src_node, target=target_node)
                # nodes_enroute include both source and target nodes
                mask[nodes_enroute] = True
            except NetworkXNoPath:
                pass
            if include_incident_en_route:
                include_endpoint_connection(data=data, mask=mask, src_node=src_node)
    edge_index, edge_attr = subgraph(subset=mask, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                     relabel_nodes=False)
    # No change since we retain the node_numbering from the original :) , we would use non-zero indices
    data.edge_index = edge_index
    data.edge_attr = edge_attr
    if save_new_dataobject:
        new_data = data.clone()
        new_data.x = data.x[mask]
        # we are relabeling the nodes here. Maybe this can be improved
        selected_nodes = torch.nonzero(mask, as_tuple=False)[:, 0].data
        # Relabel edges and everything starts from the index 0 now.
        edge_index, edge_attr = _relabel_edge_indices(edge_index=edge_index, edge_attr=edge_attr,
                                                      selected_nodes=selected_nodes)
        new_data.edge_index, new_data.edge_attr = edge_index, edge_attr
        refined_mask = torch.zeros(new_data.x.size(0), dtype=torch.bool)
        new_node_num_to_old_node_num_map = {idx: loc.item() for idx, loc in enumerate(selected_nodes)}
        old_node_num_to_new_node_num_map = {v: k for k, v in new_node_num_to_old_node_num_map.items()}
        for idx, loc in enumerate(selected_nodes):
            if loc in cuid_list:
                refined_mask[idx] = True
        new_data.mask = refined_mask
        torch.save(new_data, os.path.join(PROJECT_ROOT_DIR, 'dataset', save_filename))
        # This is for plotting the refined graph and just ensuring things are in place.
        new_data.x = torch.as_tensor(list(range(new_data.x.size(0))), dtype=torch.int)
        m_graph, data = get_nx_graph(load_edge_weights=False, prune_nodes=False, data=new_data)
        edge_list = m_graph.edges()
        new_node_to_old_node_map = {node: new_node_num_to_old_node_num_map[node] for node in m_graph.nodes()}
        new_node_2_cuid_map = {x: label_map[node] for x, node in new_node_to_old_node_map.items()}
        # For chexpert label map, key should be node_num mapped to original cuid and value is the new_node num for the entry
        chexpert_labels_map = {x: old_node_num_to_new_node_num_map[x] for x in cuid_list}
        plot_3d_graph(chexpert_labels_map, edge_list, m_graph, new_node_2_cuid_map, debug=False)
        if save_csv_file:
            rel_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper_rel.pkl'), 'rb'))
            triples = []
            for (sub, obj), rel in zip(edge_index.T, edge_attr):
                src_cuid = label_map[new_node_num_to_old_node_num_map[sub.item()]]
                rel_name = rel_map[rel.item()]
                obj_cuid = label_map[new_node_num_to_old_node_num_map[obj.item()]]
                triples.append([cuid_2_common_name[src_cuid], rel_name, cuid_2_common_name[obj_cuid]])
            with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', "relationships.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["src_cuid", "rela", "target_cuid"])
                writer.writerows(triples)
    else:
        # This is needed in order to plot the graph properly. Hence, we can remove the node feature information
        data.x = torch.nonzero(mask, as_tuple=False)[:, 0]
        print("prunning nodes based on shortest path and plotting the graph")
        chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, _ = bootstrap_data(
            load_edge_weights=False, prune_nodes=prune_nodes, data=data)
        remove = [node for node, degree in dict(m_graph.degree()).items() if degree == 0]
        m_graph.remove_nodes_from(remove)
        plot_3d_graph(chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, debug=False)


def include_endpoint_connection(data, mask, src_node):
    # additional_nodes_forward, did_match = find_nodes_sharing_an_edge(data=data, src_node=src_node, dim=1)
    # if did_match:
    #     mask[additional_nodes_forward] = True
    additional_nodes_inverse, did_match = find_nodes_sharing_an_edge(data=data, src_node=src_node, dim=0)
    if did_match:
        mask[additional_nodes_inverse] = True


def find_nodes_sharing_an_edge(data, src_node, dim):
    """
    This function finds all the edges that are connected to the given source node. Next, it would return the
    nodes that are on the other end of these edges and thereby, preserving the edge connectivity of the nodes.
    :param dim: indicates whether we are looking at source nodes or target nodes
    :param data: Entire PyG graph data object
    :param src_node: Node to check connectivity with
    :return: Matching nodes and boolean showing if there was any match
    """
    try:
        check_dim = 1 - dim
        node_locations = torch.nonzero(data.edge_index.T[:, check_dim] == src_node, as_tuple=False)[:, 0]
        matching_nodes = data.edge_index.T[node_locations, dim]
        return matching_nodes, True
    except Exception:
        return None, False


def pretty_print_path_nodes(networkx_graph, cuid_list, node_2_cuid_map):
    labels = map_cuid_2_names(important_nodes=node_2_cuid_map)
    for i, cuid in enumerate(cuid_list):
        for j in range(len(cuid_list)):
            src_node, target_node = cuid, cuid_list[j]
            visited_nodes = shortest_path(networkx_graph, source=src_node, target=target_node)
            print(f"For {labels[src_node]} to {labels[target_node]} we traverse {[labels[x] for x in visited_nodes]}")


def convert_coo_2_adj(pytorch_geo_dataobject):
    nodes = pytorch_geo_dataobject.x
    edges = pytorch_geo_dataobject.edge_index
    adj_matrix = np.zeros((nodes.size(0), nodes.size(0)))
    for src, dest in edges.t():
        adj_matrix[src.item(), dest.item()] += 1
    print(adj_matrix)


def plot_deg_dist(pytorch_geo_dataobject):
    deg = degree(pytorch_geo_dataobject.edge_index[1], pytorch_geo_dataobject.num_nodes)
    deg = sorted(deg, reverse=True)
    y_axis = np.log(deg)
    x_axis = np.log(range(len(deg)))
    plt.plot(x_axis, y_axis)
    plt.xlabel("log # nodes")
    plt.ylabel("log degree")
    plt.show()


def check_graph_stats(pytorch_geo_dataobject):
    print(f"Contains Self loop: {contains_self_loops(pytorch_geo_dataobject.edge_index)}")
    print(f"Contains Isolated Node: {contains_isolated_nodes(pytorch_geo_dataobject.edge_index)}")
    print(f"Is undirected?: {is_undirected(pytorch_geo_dataobject.edge_index)}")
    plot_deg_dist(pytorch_geo_dataobject=pytorch_geo_dataobject)


if __name__ == '__main__':
    # create_giant_dict()
    prune_nodes = False
    chexpert_labels_map, _, m_graph, node_2_cuid_map, pytorch_geo_dataobject = bootstrap_data(
        load_edge_weights=False, prune_nodes=prune_nodes)
    # save_cuid_cuid_rel()
    # check_graph_stats(pytorch_geo_dataobject=pytorch_geo_dataobject)
    # convert_coo_2_adj(pytorch_geo_dataobject=pytorch_geo_dataobject)
    cuid_cuid_shortest_path(networkx_graph=m_graph, cuid_list=list(chexpert_labels_map.keys()),
                            store_path_prob_matrix=False)
    # pretty_print_path_nodes(networkx_graph=m_graph, cuid_list=list(chexpert_labels_map.keys()),
    #                         node_2_cuid_map=node_2_cuid_map)
    save_new_dataobject = True
    trim_nodes_based_on_shortest_path(networkx_graph=m_graph, cuid_list=list(chexpert_labels_map.keys()),
                                      prune_nodes=prune_nodes, save_new_dataobject=save_new_dataobject,
                                      include_incident_en_route=False, save_filename='new_graph_big.pth',
                                      save_csv_file=save_new_dataobject)

    # Almost a completely different trend
    # print("Prunning nodes for better visualization")
    # chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, data = bootstrap_data(load_edge_weights=False,
    #                                                                                       prune_nodes=False)
    # plot_3d_graph(chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, debug=False)
