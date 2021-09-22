import os

import numpy as np
import gensim.downloader as downloader
from gensim.models import KeyedVectors

from environment_setup import PROJECT_ROOT_DIR
import nlu


import os

from models.our_method.umls_generation.language_embed.try_bert import BERTEmbedModel

os.environ['PYSPARK_PYTHON'] = '/home/chinmayp/setup/pytorch/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/home/chinmayp/setup/pytorch/bin/python'


bert_model = BERTEmbedModel()

class WordEmbeddingModel:
    def __init__(self):
        pass

    @staticmethod
    def get_fast_text_embeddings(node_names):
        model = downloader.load("fasttext-wiki-news-subwords-300")  # Same resource location as mentioned in the baseline model
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
    def get_bert_text_embeddings(node_names):
        vector_repr_dict = dict()
        for disease in node_names:
            vector_repr_dict[disease] = bert_model.get_embeds(disease)
        return vector_repr_dict


    @staticmethod
    def get_word_embed(model, word):
        try:
            # return model.get_vector(word, norm=True)
            return model[word]
        except KeyError:
            print(f"WARNING: no embeddings found for {word}, skipping!!!")
            # return model.get_vector("unk", norm=True)
            return None



    @staticmethod
    def get_embedding(node_names, strategy="fast"):
        if strategy == "fast":
            return WordEmbeddingModel.get_fast_text_embeddings(node_names=node_names)
        elif strategy == "glove":
            return WordEmbeddingModel.get_glove_embeddings(node_names=node_names)
        elif strategy == "bio":
            return WordEmbeddingModel.get_bio_embedding(node_names=node_names)
        elif strategy == 'bert':
            return WordEmbeddingModel.get_bert_text_embeddings(node_names=node_names)
        else:
            raise AttributeError("Invalid embedding strategy selected")

    @staticmethod
    def get_bio_embedding(node_names):
        filename = os.path.join(PROJECT_ROOT_DIR, "umls_extraction", "language_embed", "bio200.vec.bin")
        model = KeyedVectors.load_word2vec_format(filename, binary=True)
        vector_repr_dict = dict()
        for disease in node_names:
            words = disease.lower().split()
            final_embed = []
            for word in words:
                bio_embed_for_word = WordEmbeddingModel.get_word_embed(model, word)
                if bio_embed_for_word is not None:
                    final_embed.append(bio_embed_for_word)
            # If no match found for the entire disease name, it implies that we essentially have a situation in which
            #  the node name is most likely invalid. Hence, we should ignore it and consider it 'noise'
            if len(final_embed) == 0:
                print(f"Found an invalid combination {disease}")
                raise AttributeError("The disease name should have been pruned")
            vector_repr_dict[disease] = np.mean(final_embed, axis=0)
        return vector_repr_dict
