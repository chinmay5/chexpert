from collections import defaultdict

import os

import csv
import pickle
import time
from pprint import pprint

from tqdm import tqdm
import numpy as np

from environment_setup import PROJECT_ROOT_DIR

THRESHOLD_VAL = 10

cuid_map = {"pneumonia": "C0032285",
            "edema": "C0034063",
            "cardiomegaly": "C0018800",
            "lesion of lung": "C0577916",
            "lung opacity": "C4728208",
            "lung consolidation": "C0521530",
            "atelectasis": "C0004144",
            "pneumothorax": "C0032326",
            "fracture": "C0016658",
            "supportdevice": "C0183683"
            }

class LabelCounter:

    def __init__(self, label_list=None, select_all_cuids=False, cuid_map=None):
        assert label_list is None or cuid_map is None, "Only works for label search or direct cuid_map"

        self.label_map = defaultdict(lambda: defaultdict(int))  # key = label, value = dict{cuid: count}
        self.label_list = label_list
        self.select_all_cuids = select_all_cuids
        self.final_cuid_match = defaultdict(list)
        self.cuid_map = cuid_map
        self.conc2id = {}
        self.rel2id = {}
        self._be_smart()

    def _self_generate(self):
        with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRCONSO.RRF')) as file:
            # Pass through outer loop once and check values in the inner
            reader = csv.reader(file, delimiter='|')
            for row in tqdm(reader, desc="reading"):
                for label in self.label_list:
                    # C2910009|ENG|S|L10895526|PF|S13555892|Y|A21018258||330173||MEDCIN|SY|330173|atelectasis without respiratory distress syndrome|3|N||
                    if label in row[14].lower():
                        # Get the label back.
                        # Look at the commented line for the logic
                        # cuid_dict_of_label = self.label_list[label]
                        # cuid_dict_of_label[row[0]] = cuid_dict_of_label.get(row[0], 0) + 1
                        self.label_map[label][row[0]] += 1

    def _select_best(self):
        if self.select_all_cuids:
            for label, value_dict in self.label_map.items():
                for cuid, count in value_dict.items():
                    if count > THRESHOLD_VAL:
                        self.final_cuid_match[label].append(cuid)
            self.final_cuid_match = dict(self.final_cuid_match)
            return
        # Select the cuid which is highest count
        for label, value_dict in self.label_map.items():
            max_cuid, best_cuid = 0, None
            for cuid, count in value_dict.items():
                if count > max_cuid:
                    max_cuid = count
                    best_cuid = cuid
            # Now store this value in our code.
            self.final_cuid_match[label] = best_cuid


    def _build_parent_child_relations(self):
        print("Building relations!!!")
        triples = set()
        # valid_cuids = self.final_cuid_match.values() if not self.select_all_cuids else [k for m in self.final_cuid_match.values() for k, v in m.items()]
        if self.select_all_cuids:
            valid_cuids = []
            for _, value in self.final_cuid_match.items():
                valid_cuids.extend(value)
        else:
            valid_cuids = self.final_cuid_match.values()

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

        with open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'umls', 'META', 'MRREL.RRF')) as file:
            reader = csv.reader(file, delimiter='|')
            for row in tqdm(reader):
                if len(row[0]) == 0 or len(row[3]) == 0 or len(row[4]) == 0:
                    # empty entires, we should skip
                    continue
                # CUI1 | AUI1 | STYPE1 | REL | CUI2 | AUI2 | STYPE2 | RELA | RUI | SRUI | SAB | SL | RG | DIR | SUPPRESS |CVF
                # REL is defined as:- What relationship CUI2 has with CUI1
                for cuid in valid_cuids:
                    # PAR -> PARENT, CHD -> CHILD
                    # if cuid in [row[0], row[4]] and row[3] in ['PAR']:
                    if cuid in [row[0]] and row[3] in ['PAR']:
                        sid = add_concept(row[0], is_concept=True)
                        rid = add_concept(row[7], is_concept=False)
                        oid = add_concept(row[4], is_concept=True)
                        triples.add((sid, rid, oid))
        subjs, rels, objs = zip(*triples)
        snp = np.asarray(subjs, dtype=np.int32)
        rnp = np.asarray(rels, dtype=np.int32)
        onp = np.asarray(objs, dtype=np.int32)
        np.savez_compressed(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'graph_data.npz'),
                            subj=snp,
                            rel=rnp,
                            obj=onp)

    def _be_smart(self):
        if self.cuid_map is None:
            self._self_generate()
            self._select_best()
            pprint(self.final_cuid_match)
            pickle.dump(self.final_cuid_match, open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'label_cuid_map.pkl'), 'wb'))
        else:
            self.final_cuid_match = cuid_map
        # The relationships are built for every cuid map entry.
        self._build_parent_child_relations()
        id2conc = {v: k for k, v in self.conc2id.items()}
        pickle.dump(id2conc, open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper.pkl'), 'wb'))
        # Same for relations
        rel2conc = {v: k for k, v in self.rel2id.items()}
        pickle.dump(rel2conc, open(os.path.join(PROJECT_ROOT_DIR, 'dataset', 'mapper_rel.pkl'), 'wb'))
        print("Task Failed Successfully!!!")


if __name__ == '__main__':
    start_time = time.time()
    # Please make your life easier and pass labels in small case
    # label_list = ['Atelectasis', 'pneumonia']
    # making sure labels are in lower case.
    # label_list = list(map(lambda x: x.lower(), label_list))

    LabelCounter(label_list=None, select_all_cuids=False, cuid_map=cuid_map)
    print(f"Time taken is {time.time() - start_time} seconds")
