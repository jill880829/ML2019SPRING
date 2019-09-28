# functions for cutting sentences into tokens

__all__ = ['jieba_cut']

# ====================================

import os
import re
from collections import Counter
from multiprocessing import Pool

import jieba

# ====================================

def preprocess(sentence):
    # make all english characters in the sentence be all lowercase
    sentence = sentence.lower()

    # replace "臺" with "台"
    sentence = re.sub(r'臺', '台', sentence)

    # replace "18" with "十八"
    sentence = re.sub(r'(?<!\d)18(?!\d)', '十八', sentence)

    # replace "%" with "趴"
    sentence = re.sub(r'%|％', '趴', sentence)
    
    # replace all english except "ecfa" with " eng " (english)
    sentence = re.sub(r'ecfa', '艾克法', sentence) # replace "ecfa" with "艾克法" first
    sentence = re.sub(r'[a-z]+', ' eng ', sentence)
    sentence = re.sub(r'艾克法', 'ecfa', sentence)

    # replace "number + 趴" with " pec " (percentage)
    sentence = re.sub(r'\d+(.\d+)?趴', ' pec ', sentence)

    # replace "chiness number with length greater than two + 趴" except "十八趴" with " pec "
    sentence = re.sub(r'([一二三四五六七八九]|十(?!八))[一二三四五六七八九十]{1,}趴', ' pec ', sentence)

    # replace "number" with " num " (number)
    sentence = re.sub(r'\d+(.\d+)?', ' num ', sentence)

    # convert chiness number with length greater than two except "十八" into " num "
    sentence = re.sub(r'([一二三四五六七八九]|十(?!八))[一二三四五六七八九十]{1,}', ' num ', sentence)

    # remove all symbols and punctuations except special tokens
    sentence = re.sub(r'[^\w\s]+', ' ', sentence)

    return sentence

def _single_jieba_cut(sentence):
    sentence = preprocess(sentence)
    seg_list = list(jieba.cut(sentence))
    sentence = ' '.join([token for token in seg_list if not token.isspace()])

    return sentence

def jieba_cut(sentences, aux_data_folder = '../auxiliary_data/', n_jobs = 4, min_count = 25):
    '''
    Use jeiba to cut sentences into tokens, remove all whitespaces,
    and use single whitespace to join them into new sentences.
    This function use multicore to accelerate.

    # Arguments:
        sentences(list of str): 
        aux_data_folder(str): The auxiliary data folder which contains "dict.txt.big"
        n_jobs(int): The number of core used to accelerate.
        min_count(int)

    # Returns:
        joined_tokens(list of str):
            A list contains the cut sentences.
            A cut sentence is a string composed of tokens
            joined by single whitespace.

    # Example:
        >>> import utility
        >>> queries = utility.io.load_queries()
        >>> queries_cut = utility.cut_sentences.jieba_cut(queries)
        >>> len(queries)
        20
        >>> len(queries_cut)
        20
        >>> queries[0]
        '通姦在刑法上應該除罪化'
        >>> queries_cut[0]
        '通姦 在 刑法 上 應該 除罪 化'
    '''
    print('Cut sentences...')

    jieba.load_userdict(os.path.join(aux_data_folder, 'dict.txt.big'))
    jieba.load_userdict(os.path.join(aux_data_folder, 'my_dict.txt'))
    with Pool(processes = n_jobs) as pool:
        tokens = pool.map(_single_jieba_cut, sentences)

    # posprocess(filter min count and stop words)
    tokens = [x.split() for x in tokens]
    counter = Counter()
    for L in tokens:
        counter.update(L)
    stop_words = set()
    with open(os.path.join(aux_data_folder, 'stop_words.txt'), 'r') as fin:
        for line in fin:
            stop_words.add(line.strip())
    token_set = set()
    for k, v in counter.items():
        if v > min_count and k not in stop_words:
            token_set.add(k)
    tokens = [' '.join([x for x in L if x in token_set]) for L in tokens]

    return tokens