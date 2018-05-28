#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A template training code using LinearSVC for demenstration
"""

import pickle
import json
import os

import pandas as pd
import util
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF


DIM = 5000


class Model(object):    # TODO: rename the class name
    """ A template training code using LinearSVC for demenstration """

    def __init__(self):     # TODO: add arguments
        pass

    def train(self,train_fname):
        """ load training data from local file and train the model """
        facts, accu_label, article_label, imprison_label = self.load_data(train_fname)  # all in `pandas.Series` form
        tfidf = self.train_tfidf(facts)
        train_vector = tfidf.transform(facts)

        # learn models
        # accu_model = get_model(train_vector, accu_label)
        article_model = self.get_model(train_vector, article_label)
        # imprison_model = get_model(train_vector, imprison_label)

        # dump models
        joblib.dump(tfidf, util.TFIDF_LOC)
        # joblib.dump(accu_model, util.ACCU_LOC)
        joblib.dump(article_model, util.ART_LOC)
        # joblib.dump(imprison_model, util.IMPRISON_LOC)


    def get_model(self, train_vector, label):
        model = LinearSVC()
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
                lineDict = json.loads(line, encoding="utf-8")

                fact = lineDict["fact"]

                # return the first one of multi-accusations and multi-articles
                accu = lineDict["meta"]["accusation"][0]
                article = lineDict["meta"]["relevant_articles"][0]

                if lineDict["meta"]["term_of_imprisonment"]["death_penalty"]:
                    imprison = util.DEATH_IMPRISONMENT
                elif lineDict["meta"]["term_of_imprisonment"]["life_imprisonment"]:
                    imprison = util.LIFE_IMPRISONMENT
                else:
                    imprison = lineDict["meta"]["term_of_imprisonment"]["imprisonment"]

                facts.append(fact)

                accu_label.append(accu)
                article_label.append(article)
                imprison_label.append(imprison)

                line = f.readline()

        facts = self.cut_all(facts)

        accu_label = pd.Series(accu_label)
        article_label = pd.Series(article_label)
        imprison_label = pd.Series(imprison_label)

        if util.DUMP:
            self.dump_processed_data_to_file(facts, accu_label, article_label, imprison_label)

        return facts, accu_label, article_label, imprison_label

    def dump_processed_data_to_file(self, facts, accu_label, article_label, imprison_label):
        """ dump processed data to `.pkl` file """
        data = [facts, accu_label, article_label, imprison_label]
        with open("./mid-data.pkl", "wb") as f:
            pickle.dump(data, f)

    def load_processed_data_from_file(self):
        """ load processed data from `.pkl` file """
        with open("./mid-data.pkl", "rb") as f:
            data = pickle.load(f)
        return data

    def cut_all(self, facts):
        """ cut all lines using `jieba` """
        facts = pd.Series(facts)

        return facts.apply(util.cut_line)



if __name__ == '__main__':
    model = Model()
    model.train(os.path.join(util.DATA_DIR, util.TRAIN_FNAME))
