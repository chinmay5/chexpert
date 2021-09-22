import pickle
import re
from collections import defaultdict

import csv

import numpy as np
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, is_undirected
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR
from models.language_processing.embed_cache import EmbedCache

cache = EmbedCache(strategy="bio")

# Get the embeddings for the nodes from GloVe. We need to sort the cuid names to ensure correct mapping


cuid_map = {
    "no finding": "C0332442",
    "enlarged cardiomediastinum": "C2021206",
    "cardiomegaly": "C0018800",
    "lung opacity": "C4728208",
    "lesion of lung": "C0577916",
    "edema": "C0034063",
    "consolidation": "C0521530",  # consolidation => Lung consolidation (C0521530) or Consolidation (C0702116)
    "pneumonia": "C0032285",
    "atelectasis": "C0004144",
    "pneumothorax": "C0032326",
    "pleural effusion": "C0032227",
    "pleural others": "C0032226",
    "fracture": "C0016658",
    "support device": "C0183683"
}

class CuidInfo:
    def __init__(self, cuid, rel, rela):
        self.cuid = cuid
        self.rel = rel
        self.rela = rela

    def __hash__(self):
        return hash(self.cuid)

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return self.cuid == other.cuid
        print("Something is wrong")
        return False


list_of_semantic_types_to_ignore = [
    'Chemical',
    'Carbohydrate Sequence',
    'Nucleotide Sequence',
    'Governmental or Regulatory Activity',
    'Bird',
    'Enzyme',
    'Regulation or Law',
    'Event',
    'Geographic Area',
    'Organization',
    'Amino Acid Sequence',
    'Amino Acid, Peptide, or Protein',
    'Inorganic Chemical',
    'Nucleic Acid, Nucleoside, or Nucleotide',
    'Language',
    'Intellectual Product'
]

# def find_important_cuids():
#     semantic_types = set()
#     with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRSTY.RRF')) as file:
#         reader = csv.reader(file, delimiter='|')
#         for row in tqdm(reader, desc="reading"):
#             # C0000084|T123|A1.4.1.1.3|Biologically Active Substance|AT17597318|256|
#             semantic_types.add(row[3])
#     with open(os.path.join(PROJECT_ROOT_DIR, "dataset",'sample.txt'), 'w') as file:
#         for item in semantic_types:
#             file.write(item + '\n')

def find_important_cuids():
    invalid_cuids = set()
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRSTY.RRF')) as file:
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader, desc="reading"):
            # C0000084|T123|A1.4.1.1.3|Biologically Active Substance|AT17597318|256|
            if row[3] in list_of_semantic_types_to_ignore:
                invalid_cuids.add(row[0])
    pickle.dump(invalid_cuids, open(os.path.join(PROJECT_ROOT_DIR, "dataset", "semantic_set.pkl"), "wb"))


def create_giant_dict():
    # select * from MRCONSO where
    # TS='P' and
    # STT='PF' and
    # ISPREF='Y' and
    # LAT='ENG'
    labels = {}
    invalid_cuids = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "semantic_set.pkl"), "rb"))
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRCONSO.RRF')) as file:
        # Pass through outer loop once and check values in the inner
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader, desc="reading"):
            # Remove the labels which do not contribute much information
            if row[11] == 'SRC':
                continue
            # Also, if the given string literal turns out to contain only numbers, not very useful for us
            if len(regex_match_condition(row[-5])) == 0:
                continue
            if row[0] in invalid_cuids:
                continue
            if all([row[1] == 'ENG', row[6] == 'Y', row[4] == 'PF', row[2] == 'P']):
                labels[row[0]] = row[-5]
    pickle.dump(labels, open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "wb"))


def construct_adjacency(cuid_chexpert_target_labels, add_inverse_edge=False):
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
    # Add the mask for the Chexpert nodes
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
    node_mask = torch.zeros(len(nodes), dtype=torch.bool)
    node_mask[cuid_chexpert_target_labels] = True
    #  The list ordering is important since that is how it is perceived in pytorch_geometric
    return edge_index, edge_type, sorted(list(nodes)), node_mask


def convert_to_pygeo(edge_index, edge_type, nodes, node_mask, make_undirected=False):
    x = torch.as_tensor(nodes, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_type)
    data.mask = node_mask
    if make_undirected:
        print("making the graph undirected")
        data.edge_index, data.edge_attr = to_undirected(edge_index=data.edge_index, edge_attr=data.edge_attr)
    return data


def map_cuid_2_names(important_nodes):
    labels_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "rb"))
    labels = {}
    for node, cuid in tqdm(important_nodes.items()):
        labels[node] = labels_dict[cuid]
    return labels


def generate_dataset(make_undirected):
    # Step 0
    label_map = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper.pkl'), 'rb'))
    cuid2node_id = {v: k for k, v in label_map.items()}
    cuid_chexpert_target_labels = [cuid2node_id[cuid] for cuid in cuid_map.values()]
    edge_index, edge_type, nodes, node_mask = construct_adjacency(
        cuid_chexpert_target_labels=cuid_chexpert_target_labels)
    # Step 1
    common_name_map = map_cuid_2_names(label_map)
    node_names = [y for _, y in
                  sorted(common_name_map.items())]  # nodes are sorted to order from (0,1,2...) to map edge indices
    new_node_names = []
    for x in node_names:
        new_node_names.append(regex_match_condition(x))
    node_embeddings = cache.get_embedding(node_names=new_node_names)
    # NOTE: Initializing the node embeddings with random tensors to compute the effect
    data = convert_to_pygeo(edge_index=edge_index, edge_type=edge_type, nodes=node_embeddings,
                            node_mask=node_mask, make_undirected=make_undirected)
    print(f"Is the graph undirected ?: {is_undirected(data.edge_index)}")
    torch.save(data, os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph.pth'))
    return data


def regex_match_condition(x):
    x = re.sub(r'\[.{0,2}\]', '', x)
    x = re.sub(r'[#_\[\]]', '', x)
    x = re.sub(r'[0-9\(\)\/:;,-]', ' ', x)
    x = re.sub(r"\b[a-zA-Z]\b", "", x)
    x = x.strip()
    return x


if __name__ == '__main__':
    find_important_cuids()
    create_giant_dict()

