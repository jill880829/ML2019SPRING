# type "python jieba_whoosh.py {output_path}" to execute

import os
import sys
import csv

import numpy as np

from whoosh.index import create_in,open_dir
from whoosh.fields import *
from whoosh.query import *
from whoosh.qparser import QueryParser

import jieba
import jieba.analyse
from jieba.analyse.analyzer import ChineseAnalyzer

import utility

# =====================================

def TERM(arg):
    return Term('content', arg)

def NOT(arg):
    if type(arg) == str:
        return Not(TERM(arg))
    else:
        return Not(arg)

def OR(*args):
    L = list()
    for arg in args:
        if type(arg) == str:
            L.append(TERM(arg))
        else:
            L.append(arg)
    return Or(L)

def AND(*args):
    L = list()
    for arg in args:
        if type(arg) == str:
            L.append(TERM(arg))
        else:
            L.append(arg)
    return And(L)

# =====================================

QUERIES = [
    OR('通姦', '通姦除罪化'),
    AND('機車', OR('二段式', '兩段式')),
    AND('博弈', NOT(OR('反賭博合法化聯盟', '反賭聯盟', '反彈'))),
    AND(OR('中華航空', '華航'), '空服員', '罷工', NOT(OR('戴佐敏', '受損', '損失', '損及', '損害'))),
    AND('性交易', '合法'),
    AND(OR('ECFA', 'ecfa'), '早收清單'),
    OR(AND('證所稅', '證交稅'), '停徵'),
    AND('觀塘', '天然氣'),
    AND(OR('陸生', '中國學生'), '健保', NOT('二代')),
    AND('學生', OR('襪子', '服儀', '服裝'), NOT(OR('黃勝賢', '傳統', '張金龍'))),
    AND(OR('關閉','禁止','禁令','警告', '停止', '威脅', '反對', '暫停', '中止', '終止'),OR('交易所','虛擬貨幣','加密貨幣', '數位貨幣', AND('加密', '幣'), AND('比特', '幣'))),
    OR(AND(OR('反對', '抗議', '暫'),'學雜費','調漲'),AND('學雜費','調降'),'反教育商品化'),
    AND(OR('前瞻基礎建設','前瞻計畫'),NOT(OR('質疑','反對'))),
    AND('電競',OR('體育','運動')),
    AND(OR('抗議','自救會' ),OR('南鐵東移','鐵路東移','鐵路地下化')),
    AND('保外就醫', OR('陳水扁', '阿扁'), OR('呼籲', '人權')),
    AND(OR('18趴', '18%', '十八%', '十八趴', '年金'), '軍公教', OR('優存', '優惠存款')),
    AND('動物', '實驗', NOT(OR('犧牲', '安樂死'))),
    AND('油價', OR('凍漲', '緩漲')),
    OR(AND(OR('旺中', '旺中案'), NOT('支持')), AND(OR('旺中', '旺中案'), OR('反對', '黃國昌')))
    ]

# =====================================

jieba.enable_parallel(4)
jieba.load_userdict('./auxiliary_data/dict.txt.big')
jieba.load_userdict('./auxiliary_data/my_dict.txt')
jieba.analyse.set_stop_words('./auxiliary_data/stop_words.txt')
analyzer = ChineseAnalyzer()

schema = Schema(title = TEXT(stored = True),
                content = TEXT(stored = True, analyzer = analyzer))

if not os.path.exists('./indexdir/'):
    os.mkdir('./indexdir/')
    ix = create_in('./indexdir/', schema)
    # add documents
    news = utility.io.load_news()
    writer = ix.writer()
    print('Add documents...')
    for i, x in enumerate(news):
        if i % 1000 == 0:
            print('\t%d documents have been added.' % i)
        writer.add_document(title = 'news_%06d'%(i+1), content = x)
    writer.commit()
else:
    print('Directly open previous indexed directory...')
    ix = open_dir('./indexdir')

print('Searching...')
parser = QueryParser('content', schema = ix.schema)
with ix.searcher() as searcher:
    queries_1 = utility.io.load_queries()
    queries_2, news_index, relevance = utility.io.load_training_data()
    td_sz = len(relevance)
    L = list()
    for idx,keyword in enumerate(QUERIES):
        ques = queries_1[idx]
        popout = []
        popout_0 = []
        popout_1 = []
        popout_2 = []
        popout_3 = []
        for j in range(td_sz):
            if queries_2[j] == ques and relevance[j] == 3 and (news_index[j] not in popout):
                popout_3.append(news_index[j])
            elif queries_2[j] == ques and relevance[j] == 2 and (news_index[j] not in popout):
                popout_2.append(news_index[j])
            elif queries_2[j] == ques and relevance[j] == 1 and (news_index[j] not in popout):
                popout_1.append(news_index[j])
            elif queries_2[j] == ques and relevance[j] == 0 and (news_index[j] not in popout):
                popout_0.append(news_index[j])
            popout.append(news_index[j])
        if len(popout)>0:
            print(popout[0])
        print("result of ", keyword)
        q = keyword
        # q = parser.parse(keyword)
        results = searcher.search(q, limit = 400)
        print(len(results))
        print(len(results.top_n))
        res = []
        cnt = 0
        i = 0
        while cnt<300-len(popout_3)-len(popout_2)-len(popout_1) and i<len(results.top_n):
            element = results[i]['title']
            i+=1
            if (element not in popout_1) and (element not in popout_2) and (element not in popout_3) and (element not in popout_0):
                res.append(element)
                cnt+=1
        # print(results[0])
        print('='*40)
        ans = popout_3+popout_2+popout_1+res+popout_0
        L.append(ans[:300])

# output result
result = np.array(L)
nb_queries, rank = result.shape
with open(sys.argv[1], 'w', newline = '') as fout:
    output = csv.writer(fout)
    output.writerow(['Query_Index'] + ['Rank_%03d' % (i + 1) for i in range(rank)])
    query_idx = np.array(['q_%02d' % (i + 1) for i in range(nb_queries)])[:, np.newaxis]
    output.writerows(np.concatenate((query_idx, result), axis = 1))