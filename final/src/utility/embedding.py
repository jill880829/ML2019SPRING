# functions for converting documents and sentences into embedding vectors
# or bag of word vectors

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim

# ===================================

def sklearn_tfidf(news, queries):
    '''
    Use sklearn.feature_extraction.text.TfidfVectorizer to convert
    documents or sentences into tfidf bag of word vectors.

    # Arguments:
        news(list of str): precut news
        queries(list of str): precut queries

    # Returns:
        matrix(scipy.sparse.csr.csr_matrix)
        feature_names(list of str)
    '''
    print('compute tfidf...')
    vectorizer = TfidfVectorizer()
    vectorizer.fit(news + queries)
    matrix = vectorizer.transform(news)
    feature_names = vectorizer.get_feature_names()

    return matrix, feature_names

def gensim_bow(news, queries):
    '''
    Gensim bag of word format.

    # Arguments:
        news(list of list of str): precut news
        queries(list of list of str): precut queries

    # Returns:
        news
        queries
        dictionary(gensim.corpora.dictionary.Dictionary)
    '''
    print('Using gensim bow...')
    dictionary = gensim.corpora.Dictionary(news + queries)
    news = [dictionary.doc2bow(x) for x in news]
    queries = [dictionary.doc2bow(x) for x in queries]

    return news, queries, dictionary

def gensim_tfidf(news, queries):
    '''
    Gensim tfidf.

    # Arguments:
        news(list of str): precut news
        queries(list of str): precut queries
        dictionary(gensim.corpora.dictionary.Dictionary)

    # Returns:
        news
        queries
    '''
    print('using gensim_tfidf...')
    news, queries, dictionary = gensim_bow(news, queries)
    tfidf = gensim.models.TfidfModel(news + queries)
    news = tfidf[news]
    queries = tfidf[queries]

    return news, queries, dictionary
