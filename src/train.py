#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A template training code using LinearSVC for demenstration

ADD: use xgboost model for task 2.
"""

import os
import json
import pickle
from utils import util
from datetime import datetime as dt

import keras
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.callbacks import CSVLogger

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
import scipy
import h5py

DIM = 5000
BATCH_SIZE = 32


import sys
original = sys.stdout
sys.stdout = open('redirect.txt', 'w')
print('This is your redirected text:')
print('time logger:')


class Model(object):
    """ A template training code using LinearSVC for demenstration """

    def __init__(self):
        self.words_num = 1
        self.max_len = 80

    def train(self, train_fname):
        """ load training data from local file and train the model """
        pre = dt.now()

        if util.LOAD:
            [facts, accu_label, article_label, imprison_label] = self.load_processed_data_from_file()
        else:
            facts, accu_label, article_label, imprison_label = self.load_data(train_fname)  # all in `pandas.Series` form

        if util.DEBUG:
            print("DEBUG: load data finished.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        tfidf = self.train_tfidf(facts)
        train_vector = tfidf.transform(facts)
        if util.DEBUG:
            print("DEBUG: make tfidf model finished.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        # for task 2 - article
        tok = keras.preprocessing.text.Tokenizer(num_words=100000)
        tok.fit_on_texts(facts)
        self.words_num = len(tok.word_index)
        print('words count:', self.words_num)
        train_vector2 = tok.texts_to_sequences(facts)

        train_vector2 = keras.preprocessing.sequence.pad_sequences(train_vector2, maxlen=self.max_len)
        # Convert labels to categorical one-hot encoding
        one_hot_labels = keras.utils.to_categorical(article_label, num_classes=183)

        if util.DEBUG:
            print("DEBUG: make task2 data finished.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        # learn models
        accu_model = self.get_model('accu', train_vector, accu_label)
        if util.DEBUG:
            print("DEBUG: accusation model learnt.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        article_model = self.get_model('article', train_vector2, one_hot_labels)
        if util.DEBUG:
            print("DEBUG: article model learnt.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        imprison_model = self.get_model('imprison', train_vector, imprison_label)
        if util.DEBUG:
            print("DEBUG: imprisonment model learnt.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

        # dump models
        joblib.dump(tfidf, util.TFIDF_LOC)
        joblib.dump(accu_model, util.ACCU_LOC)
        # joblib.dump(article_model, util.ART_LOC)
        # special saving
        joblib.dump(tok, util.TOKEN_LOC)
        article_model.save(util.ART_LOC)
        joblib.dump(imprison_model, util.IMPRISON_LOC)

        if util.DEBUG:
            print("DEBUG: models dumped.")
            print("utill now time cost:", (dt.now() - pre).seconds, 's.')

    def get_model(self, kind, train_vector, labels):
        """ train models for different tasks with  different vetors and labels """
        # model = LinearSVC()
        if kind == 'accu' or kind == 'imprison':
            model = LinearSVC()  # linear SVC classifier
            model.fit(train_vector, labels)
        else:
            print('Build model...')
            model = Sequential()
            model.add(Embedding(self.words_num+1, 128, input_length=self.max_len))
            model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(183, activation='softmax'))

            # try using different optimizers and different optimizer configs
            model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            print(model.summary())
            print('Train...')
            csv_logger = CSVLogger('log.csv', append=True, separator=';')
            model.fit(train_vector, labels,
                      batch_size=BATCH_SIZE,
                      epochs=6, callbacks=[csv_logger])
        return model

    def train_tfidf(self, facts):
        """ train the TFIDF vectorizer model """
        tfidf = TFIDF(
            min_df=5,
            max_features=DIM,
            ngram_range=(1, 3)
        )
        tfidf.fit(facts)
        if util.DEBUG:
            print("DEBUG: TF-IDF model learnt.")
        return tfidf

    def load_data(self, train_fname):
        """ load data from local file """
        facts = []

        accu_label = []
        article_label = []
        imprison_label = []

        with open(train_fname, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line_dict = json.loads(line, encoding="utf-8")

                fact = line_dict["fact"]

                accu = util.get_label(line_dict, "accu")
                article = util.get_label(line_dict, "law")
                imprison = util.get_label(line_dict, "time")

                facts.append(fact)

                accu_label.append(accu)
                article_label.append(article)
                imprison_label.append(imprison)

                line = f.readline()
        if util.DEBUG:
            print("DEBUG: training file loaded.")

        facts = pd.Series(facts)
        facts = self.cut_all(facts)

        if util.DEBUG:
            print("DEBUG: training data segmented.")

        accu_label = pd.Series(accu_label)
        article_label = pd.Series(article_label)
        imprison_label = pd.Series(imprison_label)

        if util.DUMP:
            self.dump_processed_data_to_file(facts, accu_label, article_label, imprison_label)

        return facts, accu_label, article_label, imprison_label

    def dump_processed_data_to_file(self, facts, accu_label, article_label, imprison_label):
        """ dump processed data to `.pkl` file """
        data = [facts, accu_label, article_label, imprison_label]
        with open(util.MID_DATA_PKL_FILE_LOC, "wb") as f:
            pickle.dump(data, f)
        if util.DEBUG:
            print("DEBUG: data dumped to `.pkl` file")

    def load_processed_data_from_file(self):
        """ load processed data from `.pkl` file """
        with open(util.MID_DATA_PKL_FILE_LOC, "rb") as f:
            data = pickle.load(f)
        if util.DEBUG:
            print("DEBUG: data loaded from `.pkl` file")
        return data

    def cut_all(self, facts):
        """ cut all lines using `jieba` """
        return facts.apply(util.cut_line)


if __name__ == '__main__':

    model = Model()
    model.train(os.path.join(util.DATA_DIR, util.TRAIN_FNAME))
    # model.train(os.path.join(util.DATA_DIR, util.SAMPLE_FNAME))
    if util.DEBUG:
        print("DEBUG: training finished.")



sys.stdout = original