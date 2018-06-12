#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import util
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF

class Preprocess(object):
    """ pre-process data, get a basic understanding of the data """
    def __init__(self):
        pass

    def train_tfidf(self, facts):
        tfidf = TFIDF(min_df=5, max_features=DIM, ngram_range=(1, 3))
        tfidf.fit(facts)

    def load_data(self, fname):
        """ load data from local file """
        facts = []

        accu_label = []
        article_label = []
        imprison_label = []

        with open(fname, 'r') as f:
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
            print("training file loaded.")

        facts = self.cut_all(facts)

        if util.DEBUG:
            print("training data segmented.")

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

    def cut_all(self, facts):
        """ cut all lines using `jieba` """
        facts = pd.Series(facts)

        return facts.apply(util.cut_line)

    def learn_accu_label(self, accu_label):

    def preprocess(self, fname):
        facts, accu_label, article_label, imprison_label = load_data(fname)


