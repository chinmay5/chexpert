import pickle
import re
from collections import defaultdict

import csv

import numpy as np
import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from models.language_processing.word_embedding_model import get_embedding


def construct_adjacency(add_inverse_edge=False):
    """
    Constructs the Adjacency Matrix.
    :param add_inverse_edge (bool, optional): Whether to include inverse edges or not. Default: False
    :return: edge_index, edge_type, nodes
    """
    file = np.load(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_data.npz'))

    sub, rel, obj = file['subj'], file['rel'], file['obj']

    data = [z for z in zip(sub, rel, obj)]
    num_rel = len(set(rel.tolist()))
    edge_index, edge_type = [], []

    # Also construct the different nodes that we have here
    nodes = set(sub.tolist())
    nodes.update(obj.tolist())

    for sub, rel, obj in data:
        edge_index.append((sub, obj))
        edge_type.append(rel)

    # Adding inverse edges
    if add_inverse_edge:
        for sub, rel, obj in data:
            edge_index.append((obj, sub))
            edge_type.append(rel + num_rel)

    edge_index = torch.LongTensor(edge_index).t()
    edge_type = torch.LongTensor(edge_type)
    #  The list ordering is important since that is how it is perceived in pytorch_geometric
    return edge_index, edge_type, sorted(list(nodes))


def convert_to_pygeo(edge_index, edge_type, nodes):
    x = torch.as_tensor(nodes, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_type)
    return data

def map_cuid_2_names(important_nodes):
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
            for node, cuid in important_nodes.items():
                if all([row[1] == 'ENG', row[6] == 'Y', row[4] == 'PF', row[2] == 'P', row[0] == cuid]):
                    labels[node] = row[-5]
    return labels

def generate_dataset():
    edge_index, edge_type, nodes = construct_adjacency()
    # Step 1
    # Get the embeddings for the nodes from GloVe. We need to sort the cuid names to ensure correct mapping
    label_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper.pkl'), 'rb'))
    node_to_cuid = {node: label_map[node] for node in nodes}
    common_name_map = map_cuid_2_names(node_to_cuid)
    node_names = [y for _, y in sorted(common_name_map.items())] # nodes are sorted to order from (0,1,2...) to map edge indices
    new_node_names = []
    for x in node_names:
        new_node_names.append(re.sub(r'[0-9\(\)\/:;,-]', ' ', x))
    node_embeddings = get_embedding(node_names=new_node_names, strategy="glove")
    data = convert_to_pygeo(edge_index, edge_type, node_embeddings)
    torch.save(data, os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
    return data


if __name__ == '__main__':
    generate_dataset()