import atexit
import pickle

import os

from environment_setup import PROJECT_ROOT_DIR
import numpy as np

from models.our_method.umls_generation.language_embed.word_embedding_model import WordEmbeddingModel


class EmbedCache(WordEmbeddingModel):
    def __init__(self, strategy):
        super(EmbedCache, self).__init__()
        self.word_embeds = dict()
        # If the cache embed is present, re-use it
        atexit.register(self.save_dict)
        self.strategy = strategy
        if os.path.exists(os.path.join(PROJECT_ROOT_DIR, "umls_extraction", "language_embed",
                                       f"{self.strategy}_embed_cache.pkl")):
            self.word_embeds = pickle.load(
                open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", "language_embed",
                                                    f"{self.strategy}_embed_cache.pkl"), "wb"))
            print("cache loaded!!!")

    def update_nodes(self, vector_repr_dict):
        for key, repr in vector_repr_dict.items():
            self.word_embeds[key] = repr

    def is_contained(self, query_word_list):
        query_set = set(query_word_list)
        return query_set.issubset(self.word_embeds.keys())

    def get_embedding(self, node_names, strategy=None):
        if self.is_contained(query_word_list=node_names):
            return self.get_from_cache(node_names)
        else:
            vector_repr_dict = super().get_embedding(node_names=node_names, strategy=self.strategy)
            self.update_nodes(vector_repr_dict)
        word_embeds = self.process_word_embed_dict(vector_repr_dict)
        return word_embeds

    def process_word_embed_dict(self, vector_repr_dict):
        vector_repr = [vec for _, vec in vector_repr_dict.items()]
        return np.stack(vector_repr)

    def save_dict(self):
        pickle.dump(self.word_embeds,
                    open(os.path.join(PROJECT_ROOT_DIR, "models", "our_method", "umls_generation", "language_embed",
                                      f"{self.strategy}_embed_cache.pkl"), "wb"))
        print("Saved cache to disk!!!")

    def get_from_cache(self, node_names):
        word_embeds_from_cache = [self.word_embeds[key] for key in node_names]
        return np.stack(word_embeds_from_cache)


if __name__ == '__main__':
    cache = EmbedCache(strategy="bert")
    import torch
    import numpy as np
    from models.our_method.umls_generation.utils.graph_gen_utils import cuid_map
    for idx, name in enumerate(cuid_map):
        atl_ours = torch.tensor(cache.get_embedding(node_names=[name]))
        print(f"For {name} norm of embedding is {torch.linalg.norm(atl_ours)}")

