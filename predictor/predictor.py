#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
author: feng
python version: 3.x

A template prediction code
"""
import json
import os
import util
import pandas as pd
from sklearn.externals import joblib


class Predictor(object):
    """ """
    def __init__(self):
        self.batch_size = 128

        self.tfidf_model = joblib.load(util.TFIDF_LOC)
        if util.DEBUG:
            print("TF-IDF model loaded.")

        self.article_model = joblib.load(util.ART_LOC)
        if util.DEBUG:
            print("article model loaded.")

        # self.accusation_model = joblib.load(util.ACCU_LOC)    # TODO: currently no this model
        # if util.DEBUG:
        #     print("accusation model loaded.")

        # self.imprisonment_model = joblib.load(util.IMPRISON_LOC)  # TODO: currently no this model
        # if util.DEBUG:
        #     print("imprisonment model loaded.")

    def predict_article(self, vector):
        article = self.article_model.predict(vector)
        return article

    def predict_accusation(self, vector):
        return list()     # TODO: unimplemented

    def predict_imprisonment(self, vector):
        return None     # TODO: unimplemented

    def predict(self, content):
        result = []

        facts_words = pd.Series(content).apply(util.cut_line)

        vectors = self.tfidf_model.transform(facts_words)

        for vector in vectors:
            ans = dict()

            ans["articles"] = self.predict_article(vector)  # list

            ans["accusation"] = self.predict_accusation(vector)

            ans["imprisonment"] = self.predict_imprisonment(vector)

            result.append(ans)
        return result



