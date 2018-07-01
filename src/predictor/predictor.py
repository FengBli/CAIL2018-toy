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
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from pprint import pprint
import scipy
import h5py

# TF-IDF model dumped file location
TFIDF_LOC = "./predictor/model/tfidf.model"

# accusation model dumped file location
ACCU_LOC = "./predictor/model/accusation.model"

# Keras Tokenizer model dumped file location
TOKEN_LOC = "./predictor/model/token.model"

# article model dumped file location
ART_LOC = "./predictor/model/article.h5"

# imprisonment model dumped file location
IMPRISON_LOC = "./predictor/model/imprisonment.model"

# location of user-defined dictionary file
USER_DICT_LOC = "./predictor/userdict.txt"

# location of stopwords file
STOPWORDS_LOC = "./predictor/stopwords.txt"

# print some log info or not
DEBUG = True

stopwords = None

jieba.load_userdict(USER_DICT_LOC)


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

    return text


class Predictor(object):
    """ Predictor class required for submission """

    def __init__(self):
        self.batch_size = 256

        self.tfidf_model = joblib.load(TFIDF_LOC)
        if DEBUG:
            print("DEBUG: TF-IDF model loaded.")

        self.token_model = joblib.load(TOKEN_LOC)
        if DEBUG:
            print("DEBUG:TOKEN model loaded.")

        # self.article_model = joblib.load(ART_LOC)
        self.article_model = load_model(ART_LOC)
        if DEBUG:
            print("DEBUG: article model loaded.")

        self.accusation_model = joblib.load(ACCU_LOC)
        if DEBUG:
            print("DEBUG: accusation model loaded.")

        self.imprisonment_model = joblib.load(IMPRISON_LOC)
        if DEBUG:
            print("DEBUG: imprisonment model loaded.")

    def predict_article(self, lst):
        # print('inininnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn')
        # pprint(lst)
        # print('in shape: ', lst.shape)
        article = self.article_model.predict(lst)
        # pprint(article)
        # print('art shape: ', article.shape)
        index = np.argmax(article)
        # print('index: ', index)
        return [index + 1]

    def predict_accusation(self, vector):
        accusation = self.accusation_model.predict(vector)
        return [accusation[0] + 1]

    def predict_imprisonment(self, vector):
        imprisonment = self.imprisonment_model.predict(vector)
        return imprisonment

    def predict(self, content):
        result = []

        facts_words = pd.Series(content).apply(cut_line)

        vectors = self.tfidf_model.transform(facts_words)

        lsts = self.token_model.texts_to_sequences(facts_words)
        lsts = pad_sequences(lsts, maxlen=80, value=0)

        lens = len(lsts)

        for index in range(lens):
            vector = vectors[index]
            lst = lsts[index]
            ans = dict()
            # pprint(vector)
            # print('-----------------------')
            # pprint(lst)
            # print('lst shape: ', lst.shape)
            """
            TODO
            """
            # csr = scipy.sparse.csr_matrix(vector, vector.shape)
            # dtrain = xgb.DMatrix(csr)
            # ans["articles"] = self.predict_article(dtrain)
            ans["articles"] = self.predict_article(np.array([lst]))

            ans["accusation"] = self.predict_accusation(vector)

            ans["imprisonment"] = self.predict_imprisonment(vector)

            result.append(ans)
        return result
