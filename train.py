#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A template training code using LinearSVC for demenstration
"""

import os
import json
import pickle
from utils import util

import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF


DIM = 5000


class Model(object):
    """ A template training code using LinearSVC for demenstration """

    def __init__(self):
        pass

    def train(self,train_fname):
        """ load training data from local file and train the model """
        facts, accu_label, article_label, imprison_label = self.load_data(train_fname)  # all in `pandas.Series` form
        tfidf = self.train_tfidf(facts)
        train_vector = tfidf.transform(facts)

        # learn models
        accu_model = self.get_model(train_vector, accu_label)
        if util.DEBUG:
            print("DEBUG: accusation model learnt.")

        article_model = self.get_model(train_vector, article_label)
        if util.DEBUG:
            print("DEBUG: article model learnt.")

        imprison_model = self.get_model(train_vector, imprison_label)
        if util.DEBUG:
            print("DEBUG: imprisonment model learnt.")

        # dump models
        joblib.dump(tfidf, util.TFIDF_LOC)
        joblib.dump(accu_model, util.ACCU_LOC)
        joblib.dump(article_model, util.ART_LOC)
        joblib.dump(imprison_model, util.IMPRISON_LOC)

        if util.DEBUG:
            print("DEBUG: models dumped.")


    def get_model(self, train_vector, label):
        """ train models for different tasks with  different vetors and labels """
        # model = LinearSVC()
        model = RFC()   # random forest classifier

        model.fit(train_vector, label)
        return model

    def train_tfidf(self, facts):
        """ train the TFIDF vectorizer model """
        tfidf = TFIDF(
            min_df = 5,
            max_features = DIM,
            ngram_range = (1, 3)
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

        with open(train_fname, 'r') as f:
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
    # model.train(os.path.join(util.DATA_DIR, util.TRAIN_FNAME))
    model.train(os.path.join(util.DATA_DIR, util.SAMPLE_FNAME))
    if util.DEBUG:
        print("DEBUG: training finished.")
