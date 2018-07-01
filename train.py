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
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import numpy as np
import logging
from random import shuffle

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

DIM = 5000


def list_pdSeries(pdSeries):
    # pdSeries是pandas.Series类型
    for item in pdSeries.items():
        yield item[0], item[1]


def tg_data_from_pdSeries(pdSeries, info=''):
    # 从pandas.Series数据类型提取，并转化成存储TaggedDocument的list
    tg_list = []
    for num, content in list_pdSeries(pdSeries):
        tg_info = '%s_%s' % (info, num)
        tg = TaggedDocument(content, [tg_info])
        print(num, content)
        tg_list.append(tg)
    return tg_list


class Model(object):
    """ A template training code using LinearSVC for demenstration """

    def __init__(self):
        pass

    def feature_extract(self, facts, method='tfidf'):
        if method == 'tfidf':
            tfidf = self.train_tfidf(facts)
            train_vector = tfidf.transform(facts)
            # dump models
            joblib.dump(tfidf, util.TFIDF_LOC)
            return train_vector

        elif method == 'doc2vec':
            d2v_model_dm, d2v_model_dbow = self.train_doc2vec()
            docs = tg_data_from_pdSeries(facts, info="TRAIN")
            docs_for_train = docs.copy()
            d2v_model_dm.build_vocab(docs)
            d2v_model_dbow.build_vocab(docs)
            for _ in range(10):
                print("--------------------\n", _)
                d2v_model_dm.train(documents=docs_for_train, total_examples=d2v_model_dm.corpus_count, epochs=5)
                d2v_model_dbow.train(documents=docs_for_train, total_examples=d2v_model_dbow.corpus_count, epochs=5)
                shuffle(docs_for_train)
            # d2v_model_dm.train(documents=docs, total_examples=d2v_model_dm.corpus_count, epochs=5)
            # d2v_model_dbow.train(documents=docs, total_examples=d2v_model_dbow.corpus_count, epochs=5)
            # 保存模型，并且转换成用于分类的数据格式
            d2v_model_dm.save(os.path.join(util.D2V_MODEL_PATH, 'd2v_model_dm.model'))
            d2v_model_dbow.save(os.path.join(util.D2V_MODEL_PATH, 'd2v_model_dbow.model'))
            train_vector_dm = [np.array(d2v_model_dm.docvecs[z.tags[0]]).reshape((1, d2v_model_dm.wv.vector_size))
                               for z in docs]
            train_vector_dbow = [np.array(d2v_model_dbow.docvecs[z.tags[0]]).reshape((1, d2v_model_dbow.wv.vector_size))
                                 for z in docs]
            train_vector_dm = np.concatenate(train_vector_dm)
            train_vector_dbow = np.concatenate(train_vector_dbow)
            train_vector = np.concatenate((train_vector_dm, train_vector_dbow), axis=1)
            return train_vector

        elif method == 'word2vec':
            pass
            train_vector = []
            word2vec_model = self.train_word2vec(facts)
            for _, words in list_pdSeries(facts):
                d2v = np.zeros(shape=(1, word2vec_model.wv.vector_size))
                count = 0.0
                for word in words:
                    if word in word2vec_model.wv.vocab:
                        d2v += word2vec_model.wv[word].reshape(1, -1)
                        count += 1.0
                train_vector.append(d2v/count)
            train_vector = np.concatenate(train_vector, axis=0)
            word2vec_model.save(os.path.join(util.W2V_MODEL_PATH, 'w2v_model.model'))
            print(train_vector.shape)
            return train_vector

        else:
            pass

    def train(self, train_fname):
        """ load training data from local file and train the model """
        facts, accu_label, article_label, imprison_label = self.load_data(train_fname)    # all in `pandas.Series` form

        train_vector = self.feature_extract(facts=facts, method='doc2vec')
        # d2v_model_dm, d2v_model_dbow = self.train_doc2vec()
        # docs = tg_data_from_pdSeries(facts, info="TRAIN")
        # print(docs)
        # d2v_model_dm.build_vocab(docs)
        # d2v_model_dbow.build_vocab(docs)
        # d2v_model_dm.train(documents=docs, total_examples=d2v_model_dm.corpus_count, epochs=15)
        # d2v_model_dbow.train(documents=docs, total_examples=d2v_model_dbow.corpus_count, epochs=15)
        #
        # d2v_model_dm.save(os.path.join(util.D2V_MODEL_PATH, 'd2v_model_dm.model'))
        # d2v_model_dbow.save(os.path.join(util.D2V_MODEL_PATH, 'd2v_model_dbow.model'))
        # train_vector_dm = [np.array(d2v_model_dm.docvecs[z.tags[0]]).reshape((1, 150)) for z in docs]
        # train_vector_dbow = [np.array(d2v_model_dbow.docvecs[z.tags[0]]).reshape((1, 150)) for z in docs]
        # train_vector_dm = np.concatenate(train_vector_dm)
        # train_vector_dbow = np.concatenate(train_vector_dbow)
        # train_vector = np.concatenate((train_vector_dm, train_vector_dbow), axis=1)

        if util.DEBUG:
            print("DEBUG: feature extraction finished. Followed by classification training.")

        # learn models
        accu_model = self.get_model(train_vector, accu_label)
        joblib.dump(accu_model, util.ACCU_LOC)
        if util.DEBUG:
            print("DEBUG: accusation model learnt and saved.")

        article_model = self.get_model(train_vector, article_label)
        joblib.dump(article_model, util.ART_LOC)
        if util.DEBUG:
            print("DEBUG: article model learnt and saved.")

        imprison_model = self.get_model(train_vector, imprison_label, 'RandomForestRegressor')
        joblib.dump(imprison_model, util.IMPRISON_LOC)
        if util.DEBUG:
            print("DEBUG: imprisonment model learnt and saved.")

        if util.DEBUG:
            print("DEBUG: models dumped.")

    def get_model(self, train_vector, label, model_name='linearSVC'):
        """ train models for different tasks with  different vetors and labels """
        if model_name == 'linearSVC':
           model = LinearSVC()
        elif model_name == 'RandomForestClassifier':
            model = RFC()
        elif model_name == 'RandomForestRegressor':
            model = RandomForestRegressor()
        else:
            pass
        model.fit(train_vector, label)
        return model

    def train_tfidf(self, facts):
        """ train the TFIDF vectorizer model """
        tfidf = TFIDF(min_df=5, max_features=DIM, ngram_range=(1, 3))
        tfidf.fit(facts)

        if util.DEBUG:
            print("DEBUG: TF-IDF model learnt.")
        return tfidf

    def train_word2vec(self, facts):
        w2v_model = Word2Vec(facts, size=200, window=10, min_count=5, workers=8, iter=7)

        '''
        w2v_model = Word2Vec(facts, size=300, window=10, min_count=5, workers=4)
        SCORE: [0.6830846021064603, 0.6526460728001391, 0.4629383536359931]
        w2v_model = Word2Vec(facts, size=200, window=10, min_count=10, workers=8, iter=10)
        SCORE: [0.6722528369748115, 0.6474494943839428, 0.4618370862556552]
        
        '''

        if util.DEBUG:
            print("DEBUG: word2vec model learnt and saved.")
        return w2v_model

    def train_doc2vec(self):
        d2v_model_dm = Doc2Vec(min_count=10, window=10, vector_size=150, sample=3e-5, negative=5, dm=1, workers=8)
        d2v_model_dbow = Doc2Vec(min_count=10, window=10, vector_size=150, sample=3e-5, negative=5, dm=0, workers=8)

        return d2v_model_dm, d2v_model_dbow

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
    # model.train(os.path.join(util.DATA_DIR, util.TRAIN_FNAME))
    model.train(os.path.join(util.DATA_DIR, util.TRAIN_FNAME))
    if util.DEBUG:
        print("DEBUG: training finished.")
