from collections import defaultdict

import os

import csv
import pickle
import time

from tqdm import tqdm
import numpy as np

from environment_setup import PROJECT_ROOT_DIR
from models.our_method.graph_gen_utils import generate_dataset, cuid_map, CuidInfo

english_cuid_dict = pickle.load(open(os.path.join(PROJECT_ROOT_DIR, "dataset", "giant_map.pkl"), "rb"))



def my_rediculously_huge_umls_dict_par_chd():
    # NOTE: At this point, we are not taking into account the kind of relationship in the nodes
    the_huge_connectivity_dict = defaultdict(set)
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRREL.RRF')) as file:
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader):
            if len(row[0]) == 0 or len(row[3]) == 0 or len(row[4]) == 0:
                # empty entires, we should skip
                continue
            # https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_REL
            # We go for all relatiosn except del, no-mapping and self-related
            if row[3] in ['PAR'] and all([row[0] in english_cuid_dict.keys(), row[4] in english_cuid_dict.keys()]):
                the_huge_connectivity_dict[row[0]].add(CuidInfo(cuid=row[4], rel=row[3], rela=row[7]))
    pickle.dump(the_huge_connectivity_dict,
                open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'huge_cuid_cuid_obj.pkl'), 'wb'))


def my_rediculously_huge_umls_dict_all_rel():
    # NOTE: At this point, we are not taking into account the kind of relationship in the nodes
    the_huge_connectivity_dict = defaultdict(set)
    with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRREL.RRF')) as file:
        reader = csv.reader(file, delimiter='|')
        for row in tqdm(reader):
            if len(row[0]) == 0 or len(row[3]) == 0 or len(row[4]) == 0:
                # empty entires, we should skip
                continue
            # https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus/release/abbreviations.html#mrdoc_REL
            # We go for all relations except del, no-mapping and self-related
            if row[3] in ['DEL', 'XR', 'RL']:
                continue
            #   Add relations only when they contain some meaningful semantic information
            if all([row[0] in english_cuid_dict.keys(), row[4] in english_cuid_dict.keys()]):
                the_huge_connectivity_dict[row[0]].add(CuidInfo(cuid=row[4], rel=row[3], rela=row[7]))
                # the_huge_connectivity_dict[row[0]].add(row[4])
    pickle.dump(the_huge_connectivity_dict,
                open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'huge_cuid_cuid_obj.pkl'), 'wb'))


def my_rediculously_huge_umls_dict(all_rel):
    if all_rel:
        my_rediculously_huge_umls_dict_all_rel()
    else:
        my_rediculously_huge_umls_dict_par_chd()


class LabelCounter:

    def __init__(self, cuid_map=None, num_hops=1, defer_undirected_to_pyG=False):

        self.label_map = defaultdict(lambda: defaultdict(int))  # key = label, value = dict{cuid: count}
        self.num_hops = num_hops
        self.final_cuid_match = cuid_map
        self.conc2id = {}
        self.rel2id = {}
        self.defer_undirected_to_pyG = defer_undirected_to_pyG
        self._be_smart()

    def _build_parent_child_relations(self):
        print("Building relations!!!")
        triples = set()
        valid_cuids = self.final_cuid_match.values()
        # We are going to perform K-hop neighbourhood search step. Hence, at each step, update the valid_cuids with the
        # nodes we visited in the latest traversal
        the_huge_connectivity_dict = pickle.load(
            open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'huge_cuid_cuid_obj.pkl'), 'rb'))
        print("Connectivity dict loaded!!!")
        for k in range(self.num_hops):
            new_cuids_visited = set()

            def add_concept(conc, is_concept):
                if is_concept:
                    if conc in self.conc2id:
                        cid = self.conc2id[conc]
                    else:
                        cid = len(self.conc2id)
                        self.conc2id[conc] = cid
                    return cid
                else:
                    if conc in self.rel2id:
                        rid = self.rel2id[conc]
                    else:
                        rid = len(self.rel2id)
                        self.rel2id[conc] = rid
                    return rid

            for cuid in tqdm(valid_cuids):
                # neighbourhood_cuids = the_huge_connectivity_dict[cuid]
                # new_cuids_visited.update(neighbourhood_cuids)
                cuid_info_objects = the_huge_connectivity_dict[cuid]
                neighbourhood_cuids = [cuid_info_obj.cuid for cuid_info_obj in cuid_info_objects]
                new_cuids_visited.update(neighbourhood_cuids)
                # for neigh in neighbourhood_cuids:
                for neigh_cuid_obj in cuid_info_objects:
                    neigh = neigh_cuid_obj.cuid
                    # We are not including self loops
                    if cuid == neigh:
                        continue
                    # TODO: Include different kinds of relations
                    sid = add_concept(cuid, is_concept=True)
                    rid = add_concept(neigh_cuid_obj.rela, is_concept=False)
                    oid = add_concept(neigh, is_concept=True)
                    # Include the triplets only when they are not already a part
                    if (sid, rid, oid) not in triples:
                        if self.defer_undirected_to_pyG:
                            # Check if inverse relation is already present. If so, skip it. We create it later on
                            if (oid, rid, sid) in triples:
                                # We need to skip these elements
                                continue
                        triples.add((sid, rid, oid))
                    else:
                        # print("repeated triple skipping")
                        pass
            valid_cuids = new_cuids_visited
        subjs, rels, objs = zip(*triples)
        snp = np.asarray(subjs, dtype=np.int32)
        rnp = np.asarray(rels, dtype=np.int32)
        onp = np.asarray(objs, dtype=np.int32)
        np.savez_compressed(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_data.npz'),
                            subj=snp,
                            rel=rnp,
                            obj=onp)

    def _be_smart(self):
        self._build_parent_child_relations()
        id2conc = {v: k for k, v in self.conc2id.items()}
        pickle.dump(id2conc, open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper.pkl'), 'wb'))
        # Same for relations
        rel2conc = {v: k for k, v in self.rel2id.items()}
        pickle.dump(rel2conc, open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper_rel.pkl'), 'wb'))
        print("Task Failed Successfully!!!")


if __name__ == '__main__':
    start_time = time.time()
    all_rel = False
    my_rediculously_huge_umls_dict(all_rel=all_rel)
    # Please make your life easier and pass labels in small case
    # label_list = ['Atelectasis', 'pneumonia']
    # making sure labels are in lower case.
    # label_list = list(map(lambda x: x.lower(), label_list))
    # Make undirected only when we are not using all relations
    make_undirected = not all_rel
    defer_undirected_to_pyG = make_undirected
    assert defer_undirected_to_pyG == make_undirected, "Should be the same ALWAYSSSSSS!!!!"
    LabelCounter(cuid_map=cuid_map, num_hops=3, defer_undirected_to_pyG=defer_undirected_to_pyG)
    print("Generating the pytorch geometric dataset")
    generate_dataset(make_undirected=make_undirected)
    print(f"Time taken is {time.time() - start_time} seconds")
    # In order to visualize the plots, use visualizations.graph_plits.py file
