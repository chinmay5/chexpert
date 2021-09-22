import pickle
from collections import defaultdict

import csv

import pandas as pd
import torch
import numpy as np
import os
import networkx as nx

import plotly.graph_objects as go
import plotly.io as pio
from dotmap import DotMap
from matplotlib import pyplot as plt
from networkx import shortest_path
from networkx.exception import NetworkXNoPath
from tabulate import tabulate
from tqdm import tqdm

from models.our_method.umls_generation.utils.graph_gen_utils import cuid_map

pd.set_option("display.max_rows", None, "display.max_columns", None)

# This import is needed for pickling

from torch_geometric.utils import degree, subgraph, contains_self_loops, contains_isolated_nodes, is_undirected

from environment_setup import PROJECT_ROOT_DIR, GRAPH_FILE_NAME

device = torch.device("cpu")

print(f"ATTENTION!!! We are using {GRAPH_FILE_NAME} file to build the graph. If this was not your choice, "
      f"please make necessary changes")

def _bootstrap_prunning():
    bootstrap_info = DotMap()
    bootstrap_info.label_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", 'mapper.pkl'), 'rb'))
    # /home/chinmayp/workspace/chexpert/dataset
    bootstrap_info.cuid_2_common_name = pickle.load(
        open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", "giant_map.pkl"), "rb"))
    bootstrap_info.con2id = {v: k for k, v in bootstrap_info.label_map.items()}
    bootstrap_info.cuid_chexpert_target_labels = [bootstrap_info.con2id[cuid] for cuid in cuid_map.values()]
    return bootstrap_info


def get_nx_graph(load_edge_weights=False, data=None):
    if data is None:
        data = torch.load(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", GRAPH_FILE_NAME))
        # We replace the node embeddings with just the node sequences. This works since we ensure sequential numbering
        # for the nodes all through.
        data.x = torch.as_tensor(list(range(data.x.size(0))), dtype=torch.int)
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


def map_cuid_2_names(bootstrap_info, important_nodes):
    labels = defaultdict(str)
    for node, cuid in tqdm(important_nodes.items()):
        labels[node] = bootstrap_info.cuid_2_common_name[cuid]
    return labels


def plot_3d_graph_mit_masked_isolated_nodes(bootstrap_info, category_labels_map, edge_list, m_graph, node_2_cuid_map,
                                            name_plot_using_node, edge_text=None, debug=False,
                                            out_file='sample.html'):
    # Keep track of `chexpert label` nodes since we would like to have them distinct

    if debug:
        labels = node_2_cuid_map
    else:
        labels = map_cuid_2_names(bootstrap_info, node_2_cuid_map)

    node_names = [y for _, y in sorted(labels.items())]  # nodes are sorted to order from (0,1,2...) to map edge indices
    # Assign the plot a name
    title = f"UMLS subgraph showing most important nodes for: {labels[name_plot_using_node]}"

    # We try ro use plotly to visualize the graph now

    pio.renderers.default = "browser"

    # set the argument 'with labels' to False so you have unlabeled graph
    pos = nx.spring_layout(m_graph, dim=3, seed=42)

    node_info = {}
    for node_num, node_loc in pos.items():
        node_info[node_num] = node_loc

    # we  need to create lists that contain the starting and ending coordinates of each edge.
    x_edges, y_edges, z_edges = [], [], []

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
    x_nodes = [pos[i][0] for i in pos.keys()]  # x-coordinates of nodes
    y_nodes = [pos[i][1] for i in pos.keys()]  # y-coordinates
    z_nodes = [pos[i][2] for i in pos.keys()]  # z-coordinates
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
    layout = go.Layout(title=title,
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
    for node_num, idx in category_labels_map.items():
        text = labels[node_num]
        data.append(
            go.Scatter3d(x=[node_info[node_num][0].item()],
                         y=[node_info[node_num][1].item()],
                         z=[node_info[node_num][2].item()],
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
    pio.write_html(fig, os.path.join(PROJECT_ROOT_DIR,"models", "our_method", "umls_generation", out_file))


def plot_3d_graph(bootstrap_info, category_labels_map, edge_list, m_graph, node_2_cuid_map, edge_text=None, debug=False,
                  out_file='sample.html'):
    # Keep track of `chexpert label` nodes since we would like to have them distinct

    if debug:
        labels = node_2_cuid_map
    else:
        labels = map_cuid_2_names(bootstrap_info, node_2_cuid_map)

    node_names = [y for _, y in sorted(labels.items())]  # nodes are sorted to order from (0,1,2...) to map edge indices

    # We try ro use plotly to visualize the graph now

    pio.renderers.default = "browser"

    # set the argument 'with labels' to False so you have unlabeled graph
    pos = nx.spring_layout(m_graph, dim=3, seed=42)

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
    for node_num, idx in category_labels_map.items():
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
    pio.write_html(fig, os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", out_file))


def bootstrap_data(bootstrap_info, load_edge_weights=False, data=None):
    # https://deepnote.com/@deepnote/3D-network-visualisations-using-plotly-oYxeN6UXSye_3h_ulKV2Dw
    m_graph, data = get_nx_graph(load_edge_weights=load_edge_weights, data=data)
    # Num_nodes = len(m_graph.nodes())
    edge_list = m_graph.edges()
    node_2_cuid_map = {node: bootstrap_info.label_map[node] for node in m_graph.nodes()}
    nih_label_nodes = [node_num for node_num, cuid in node_2_cuid_map.items() for node_cuids in cuid_map.values()
                       if cuid in node_cuids]
    nih_labels_map = {x: np.where(data.x.numpy() == x)[0].item() for x in nih_label_nodes}
    return nih_labels_map, edge_list, m_graph, node_2_cuid_map, data


def _tabulate_and_print(bootstrap_info, path_len, cuid_list):
    cuids = [bootstrap_info.label_map[x] for x in cuid_list]
    # /home/chinmayp/workspace/chexpert/dataset
    labels_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", "giant_map.pkl"), "rb"))
    headers = [labels_dict[x] for x in cuids]
    table = tabulate(path_len, headers, tablefmt="fancy_grid")
    print(table)
    #     Let us also create a similarity mask table
    mask_table = path_len <= 3
    np.save(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation",'mask_table'), mask_table)
    return mask_table


def cuid_cuid_shortest_path(bootstrap_info, networkx_graph, cuid_list, store_path_prob_matrix=False):
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
    print(f"Avg path length is {sum(avg_shortst_path) / (len(avg_shortst_path) - len(cuid_list))}")
    _tabulate_and_print(bootstrap_info=bootstrap_info, path_len=path_len, cuid_list=cuid_list)
    path_len.fill_diagonal_(1e-20)
    prob_matrix = torch.softmax(input=path_len, dim=1)
    # Higher softmax => larger distance. So, take an inverse of that
    prob_matrix = 1 - prob_matrix
    prob_matrix.fill_diagonal_(1.0)
    if store_path_prob_matrix:
        print("storing the path length matrix")
        np.save(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", 'graph_data_prob.npy'), prob_matrix)
    # print(path_len)


def _relabel_edge_indices(edge_index, edge_attr, selected_nodes):
    for idx, val in enumerate(selected_nodes):
        indices = torch.where(edge_index[0] == val)
        edge_index[0][indices] = idx
        # Now for the destination nodes
        indices = torch.where(edge_index[1] == val)
        edge_index[1][indices] = idx
    return edge_index, edge_attr


def trim_nodes_based_on_shortest_path(bootstrap_info, networkx_graph, cuid_list, save_new_dataobject,
                                      include_incident_en_route=False, save_filename='new_graph.pth',
                                      save_csv_file=False):
    data = torch.load(os.path.join(PROJECT_ROOT_DIR,"models", "our_method", "umls_generation", GRAPH_FILE_NAME))
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
        orig_taget_label_indices = data.target_label_indices
        # we are relabeling the nodes here. Maybe this can be improved
        selected_nodes = torch.nonzero(mask, as_tuple=False)[:, 0].data
        # Relabel edges and everything starts from the index 0 now.
        edge_index, edge_attr = _relabel_edge_indices(edge_index=edge_index, edge_attr=edge_attr,
                                                      selected_nodes=selected_nodes)
        new_data.edge_index, new_data.edge_attr = edge_index, edge_attr
        refined_mask = torch.zeros(new_data.x.size(0), dtype=torch.bool)
        new_node_num_to_old_node_num_map = {idx: loc.item() for idx, loc in enumerate(selected_nodes)}
        old_node_num_to_new_node_num_map = {v: k for k, v in new_node_num_to_old_node_num_map.items()}
        taget_label_indices = [old_node_num_to_new_node_num_map[x] for x in orig_taget_label_indices]
        new_data.target_label_indices = taget_label_indices
        torch.save(new_data, os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", save_filename))
        # This is for plotting the refined graph and just ensuring things are in place.
        new_data.x = torch.as_tensor(list(range(new_data.x.size(0))), dtype=torch.int)
        m_graph, data = get_nx_graph(load_edge_weights=False, data=new_data)
        edge_list = m_graph.edges()
        new_node_to_old_node_map = {node: new_node_num_to_old_node_num_map[node] for node in m_graph.nodes()}
        new_node_2_cuid_map = {x: bootstrap_info.label_map[node] for x, node in new_node_to_old_node_map.items()}
        # For chexpert label map, key should be node_num mapped to original cuid and value is the new_node num for the entry
        # Hotfix
        category_labels_map = {old_node_num_to_new_node_num_map[x]: old_node_num_to_new_node_num_map[x] for x in
                               cuid_list}
        plot_3d_graph(bootstrap_info, category_labels_map, edge_list, m_graph, new_node_2_cuid_map, debug=False)
        if save_csv_file:
            rel_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", 'mapper_rel.pkl'), 'rb'))
            triples = []
            for (sub, obj), rel in zip(edge_index.T, edge_attr):
                src_cuid = bootstrap_info.label_map[new_node_num_to_old_node_num_map[sub.item()]]
                rel_name = rel_map[rel.item()]
                obj_cuid = bootstrap_info.label_map[new_node_num_to_old_node_num_map[obj.item()]]
                triples.append([bootstrap_info.cuid_2_common_name[src_cuid], rel_name,
                                bootstrap_info.cuid_2_common_name[obj_cuid]])
            with open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", "relationships.csv"), "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["src_cuid", "rela", "target_cuid"])
                writer.writerows(triples)
    else:
        # This is needed in order to plot the graph properly. Hence, we can remove the node feature information
        data.x = torch.nonzero(mask, as_tuple=False)[:, 0]
        print("prunning nodes based on shortest path and plotting the graph")
        chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, _ = bootstrap_data(bootstrap_info=bootstrap_info,
                                                                                     load_edge_weights=False, data=data)
        remove = [node for node, degree in dict(m_graph.degree()).items() if degree == 0]
        m_graph.remove_nodes_from(remove)
        plot_3d_graph(bootstrap_info, chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, debug=False)


def include_endpoint_connection(data, mask, src_node):
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


def trim_nodes_and_store_cuid_graph(save_new_dataobject=True, data=None):
    bootstrap_info = _bootstrap_prunning()
    chexpert_labels_map, edge_list, m_graph, node_2_cuid_map, pytorch_geo_dataobject = bootstrap_data(
        bootstrap_info=bootstrap_info, load_edge_weights=False,
        data=data)
    # check_graph_stats(pytorch_geo_dataobject=pytorch_geo_dataobject)
    cuid_cuid_shortest_path(bootstrap_info=bootstrap_info, networkx_graph=m_graph,
                            cuid_list=list(chexpert_labels_map.keys()),
                            store_path_prob_matrix=False)
    # pretty_print_path_nodes(networkx_graph=m_graph, cuid_list=list(chexpert_labels_map.keys()),
    #                         node_2_cuid_map=node_2_cuid_map)
    if save_new_dataobject:
        trim_nodes_based_on_shortest_path(bootstrap_info=bootstrap_info, networkx_graph=m_graph,
                                          cuid_list=list(chexpert_labels_map.keys()),
                                          save_new_dataobject=save_new_dataobject, include_incident_en_route=False,
                                          save_filename='umls_small_graph.pth', save_csv_file=False)
    else:
        plot_3d_graph(bootstrap_info=bootstrap_info, category_labels_map=chexpert_labels_map, edge_list=edge_list,
                      m_graph=m_graph,
                      node_2_cuid_map=node_2_cuid_map, debug=False)
