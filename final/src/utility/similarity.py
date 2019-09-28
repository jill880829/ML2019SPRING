# functions for computing the similarity between the embedding vectors

import os

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.similarities import Similarity, WmdSimilarity

# ==============================

def tfidf_matching_score(matrix, feature_names, queries):
    '''
    For each query, directly add those feature names exists in both the
    query and the tfidf bag of word vector that each news corresponding to.

    # Arguments:
        matrix(scipy.sparse.csr.csr_matrix)
        feature_names(list of str)
        queries(list of str): precut queries

    # Returns:
        score_matrix
    '''
    score_matrix = list()
    for query in queries:
        idx = [i for i, name in enumerate(feature_names) if name in query]
        score_array = np.sum(matrix.T[idx], axis = 0).A1 # flatten the "numpy.matrix"
        score_matrix.append(score_array)

    return score_matrix

def tfidf_cosine_similarity(matrix, feature_names, queries):
    '''
    For each query, directly add those feature names exists in both the
    query and the tfidf bag of word vector that each news corresponding to.

    # Arguments:
        matrix(scipy.sparse.csr.csr_matrix)
        feature_names(list of str)
        queries(list of str): precut queries

    # Returns:
        score_matrix
    '''
    score_matrix = list()
    for query in queries:
        idx = [i for i, name in enumerate(feature_names) if name in query]
        query_vector = np.zeros(shape = (matrix.shape[1],))
        query_vector[idx] = 1
        score_array = cosine_similarity(matrix, query_vector[np.newaxis, :])
        score_matrix.append(score_array.ravel())

    return score_matrix

def gensim_similarity(news, queries, dictionary, save_folder = './processed_data/'):
    '''
    Gensim similarity

    # Arguments:
        news
        queries
        dictionary(gensim.corpora.dictionary.Dictionary)

    # Returns:
        score_matrix
    '''
    print('Compute gensim similarity...')
    index = Similarity(os.path.join(save_folder, 'gensim_similarity'),
                       news,
                       num_features = len(dictionary))
    score_matrix = index[queries]
    
    return score_matrix
