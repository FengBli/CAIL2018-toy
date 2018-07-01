#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: feng
python version: 3.x

A template prediction code
"""
import os
import re
import json
import jieba
import pandas as pd
from sklearn.externals import joblib
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np



# TF-IDF model dumped file location
TFIDF_LOC = "./predictor/model/tfidf.model"

# accusation model dumped file location
ACCU_LOC = "./predictor/model/accusation.model"

# article model dumped file location
ART_LOC = "./predictor/model/article.model"

# imprisonment model dumped file location
IMPRISON_LOC = "./predictor/model/imprisonment.model"

# location of user-defined dictionary file
USER_DICT_LOC = "./predictor/userdict.txt"

# location of stopwords file
STOPWORDS_LOC = "./predictor/stopwords.txt"

# D2V model location
D2V_MODEL_DM = "./predictor/model/d2v_model_dm.model"

# D2V model dbow location
D2V_MODEL_DBOW = "./predictor/model/d2v_model_dbow.model"

# W2V model
W2V_MODEL = "./predictor/model/w2v_model.model"

# print some log info or not
DEBUG = True

stopwords = None

jieba.load_userdict(USER_DICT_LOC)


def list_pdSeries(pdSeries):
    # pdSeries是pandas.Series类型
    for item in pdSeries.items():
        yield item[0], item[1]


def load_stopwords(stopwords_fname):
    """ load stopwords into set from local file
    duplicated in `./utils/util.py`, but needed for convenience
    """
    stopwords = set()
    with open(stopwords_fname, "r", encoding="utf-8") as f:
        for line in f.readlines():
            stopwords.add(line.strip())
    print("DEBUG: stopwords loaded.")
    return stopwords


def cut_line(line):
    """ cut the single line using `jieba`
    duplicated in `./utils/util.py`, but needed for convenience
    """

    global stopwords

    if stopwords is None:
        stopwords = load_stopwords(STOPWORDS_LOC)

    # remove the date and time
    line = re.sub(r"\d*年\d*月\d*日", "", line)
    line = re.sub(r"\d*[时|时许]", "", line)
    line = re.sub(r"\d*分", "", line)

    word_list = jieba.cut(line)

    # remove the stopwords
    words = []
    for word in word_list:
        if word not in stopwords:
            words.append(word)

    text = " ".join(words)

    # correct some results
    # merge「王」and「某某」into「王某某」
    text = re.sub(" 某某", "某某", text)

    # merge「2000」and「元」into「2000元」
    text = re.sub(" 元", "元", text)
    text = re.sub(" 余元", "元", text)

    text = re.sub("价 格", "价格", text)

    text = text.strip().split()

    return text


class Predictor(object):
    """ Predictor class required for submission """
    def __init__(self):
        self.batch_size = 256

        # self.tfidf_model = joblib.load(TFIDF_LOC)
        # if DEBUG:
        #     print("DEBUG: TF-IDF model loaded.")

        self.article_model = joblib.load(ART_LOC)
        if DEBUG:
            print("DEBUG: article model loaded.")

        self.accusation_model = joblib.load(ACCU_LOC)
        if DEBUG:
            print("DEBUG: accusation model loaded.")

        self.imprisonment_model = joblib.load(IMPRISON_LOC)
        if DEBUG:
            print("DEBUG: imprisonment model loaded.")

        '''
        doc2vec model
        '''
        self.d2v_model_dm = Doc2Vec.load(D2V_MODEL_DM)
        self.d2v_model_dbow = Doc2Vec.load(D2V_MODEL_DBOW)

        '''
        word2vec model
        '''
        self.w2v_model = Word2Vec.load(W2V_MODEL)

    def predict_article(self, vector):
        article = self.article_model.predict(vector)
        return [article[0] + 1]

    def predict_accusation(self, vector):
        accusation = self.accusation_model.predict(vector)
        return [accusation[0] + 1]

    def predict_imprisonment(self, vector):
        imprisonment = self.imprisonment_model.predict(vector)
        return imprisonment

    def predict(self, content):
        result = []
        vectors_dm = []
        vectors_dbow = []
        vectors = []
        vector = []

        facts_words = pd.Series(content).apply(cut_line)

        '''tfidf'''
        # vectors = self.tfidf_model.transform(facts_words)

        '''doc2vec'''
        d2v_model_dm = self.d2v_model_dm
        d2v_model_dbow = self.d2v_model_dbow

        '''word2vec'''
        w2v_model = self.w2v_model

        # for _, fact in list_pdSeries(facts_words):
        #     d2v = d2v_model_dm.infer_vector(fact)
        #     vectors_dm.append(d2v)
        #
        # for _, fact in list_pdSeries(facts_words):
        #     d2v = d2v_model_dbow.infer_vector(fact)
        #     vectors_dbow.append(d2v)

        for _, words in list_pdSeries(facts_words):
            d2v = np.zeros(shape=(1, w2v_model.wv.vector_size))
            count = 0.0
            for word in words:
                if word in w2v_model.wv.vocab:
                    d2v += w2v_model.wv[word].reshape(1, -1)
                    count += 1.0
            vectors.append(d2v/count)

        for vector in vectors:
            ans = dict()

            ans["articles"] = self.predict_article(vector)

            ans["accusation"] = self.predict_accusation(vector)

            ans["imprisonment"] = self.predict_imprisonment(vector)

            result.append(ans)
        return result

        # for vector_dm, vector_dbow in zip(vectors_dm, vectors_dbow):
        #     vector = np.concatenate((vector_dm, vector_dbow))
        #     vector = [vector]
        #     ans = dict()
        #
        #     ans["articles"] = self.predict_article(vector)
        #
        #     ans["accusation"] = self.predict_accusation(vector)
        #
        #     ans["imprisonment"] = self.predict_imprisonment(vector)
        #
        #     result.append(ans)
        # return result



