#!usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import jieba

# the directory where the data lies
DATA_DIR  = "../data/CAIL2018-small-data/"

# training file name
TRAIN_FNAME = "data_train.json"

# sample data file name, for testing bugs
SAMPLE_FNAME = "data_sample.json"

# testing file name
TEST_FNAME = "data_test.json"

# the location of stopwords
STOPWORDS_LOC = "./stopwords.txt"

# TF-IDF model dumped file location
TFIDF_LOC = "./predictor/model/tfidf.model"

# accusation model dumped file location
ACCU_LOC = "./predictor/model/accusation.model"

# article model dumped file location
ART_LOC = "./predictor/model/article.model"

# imprisonment model dumped file location
IMPRISON_LOC = "./predictor/model/imprisonment.model"

# dump the mid-data to local `.pkl` file or not
DUMP = False

# print something log info or not
DEBUG = True

# digitize the dealth penalty and life imprisonments
DEATH_IMPRISONMENT = 600
LIFE_IMPRISONMENT = 800

jieba.load_userdict("./dictionary/userdict.txt")    # TODO: when this `util` module is loaded, will this line be executed?


def load_stopwords(stopwords_fname):
    """ load stopwords into set from local file """
    stopwords = set()
    with open(stopwords_fname, 'r') as f:
        for line in f.readlines():
            stopwords.add(line.strip())

    return stopwords


stopwords = load_stopwords(STOPWORDS_LOC)

def cut_line(line):
    """ cut the single line using `jieba` """

    global stopwords
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
