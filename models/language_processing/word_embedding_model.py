import os

import numpy as np
import gensim.downloader as downloader
from gensim.models import KeyedVectors

from environment_setup import PROJECT_ROOT_DIR


class WordEmbeddingModel:
    def __init__(self):
        pass

    @staticmethod
    def get_fast_text_embeddings(node_names):
        model = downloader.load(
            "fasttext-wiki-news-subwords-300")  # Same resource location as mentioned in the baseline model
        vector_repr_dict = dict()

        for disease in node_names:
            words = disease.lower().split()
            vector_repr_dict[disease] = np.mean([WordEmbeddingModel.get_word_embed(model, word) for word in words], axis=0)
        return vector_repr_dict

    @staticmethod
    def get_glove_embeddings(node_names):
        model = downloader.load("glove-wiki-gigaword-300")  # Same resource location as mentioned in the baseline model
        vector_repr_dict = dict()
        for disease in node_names:
            words = disease.lower().split()
            vector_repr_dict[disease] = np.mean([WordEmbeddingModel.get_word_embed(model, word) for word in words], axis=0)
        return vector_repr_dict


    @staticmethod
    def get_word_embed(model, word):
        try:
            # return model.get_vector(word, norm=True)
            return model[word]
        except KeyError:
            print(f"WARNING: no embeddings found for {word}, using UNK instead!!!")
            return model.get_vector("unk", norm=True)

    @staticmethod
    def get_embedding(node_names, strategy="fast"):
        if strategy == "fast":
            return WordEmbeddingModel.get_fast_text_embeddings(node_names=node_names)
        elif strategy == "glove":
            return WordEmbeddingModel.get_glove_embeddings(node_names=node_names)
        elif strategy == "bio":
            return WordEmbeddingModel.get_bio_embedding(node_names=node_names)
        else:
            raise AttributeError("Invalid embedding strategy selected")

    @staticmethod
    def get_bio_embedding(node_names):
        filename = os.path.join(PROJECT_ROOT_DIR, "models", "language_processing", "bio200.vec.bin")
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
        vector_repr_dict = dict()
        for disease in node_names:
            words = disease.lower().split()
            vector_repr_dict[disease] = np.mean([WordEmbeddingModel.get_word_embed(model, word) for word in words], axis=0)
        return vector_repr_dict
