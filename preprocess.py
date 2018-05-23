# !usr/env python3
# -*- coding: utf-8 -*-

"""
author: feng
python version: 3.x

将json格式数据转化为csv格式，分隔符设置为"\x00"。
使用thulac进行分词
另外，为了便于对刑期的预测，将**死刑**和**无期徒刑**数值化，分别令为600和800（单位：月）。
注：
- 此两数值后续根据情况进行调整
- 对数据中的 "[1, 2, 3]"这样的str，可以使用eval(str)得到对应list

"""

import os
import json
import re
import thulac     # Tsinghua Chinese word segementation tool
from util import *  # DATA_DIR, list2str_unicode_version
from sklearn.feature_extraction.text import TfidfVectorizer

LIFE_IMPRISONMENT = 600
DEATH_IMPRISONMENT = 800


TITLE = u"fact\x00criminal\x00money\x00accusations\x00articles\x00imprisonment"
DATA_FORMAT = u"{}\x00{}\x00{}\x00{}\x00{}\x00{}"


class Preprocess(object):
    """ preprocessing:  word segmentation and json2csv """

    def __init__(self, data_dir, stopwords_fname):
        """
        :data_dir:  train, test, validation file dir
        :stopwords_fname:   stopwords file location
        """
        self.data_dir = data_dir
        self.thulac = thulac.thulac(seg_only=True,  # 只分词，不标注词性
                                    filt=False)      # 不使用过滤器过滤无意义词语
        # self.corpus = []
        # self.vectorizer = TfidfVectorizer()

        self.load_stopwords(stopwords_fname)


    def load_stopwords(self, fname):
        """ load stopwords from local file """
        self.stopwords_set = set()
        with open(fname, 'r') as f:
            for line in f.readlines():
                self.stopwords_set.add(line.strip())


    def json2csv(self, fname, with_title=False):
        """
        Convert .json file to .csv file, using the same filename with ".csv" replacing ".json"

        :fname:         the .json file to be converted
        :with_title:    with the head title line in the dest file or not
        """

        fcsv_name = fname.replace("json", "csv")

        fcsv = open(fcsv_name, "a+", encoding="utf-8")
        if with_title:
            fcsv.write(TITLE+BR)

        with open(fname, "r") as fjson:
            line = fjson.readline()
            while line:
                lineDict = json.loads(line, encoding="utf-8")


                fact = lineDict["fact"]     # string
                fact = self.handle(fact)    # 以空格分隔开的分词结果

                meta = lineDict["meta"]     # dict

                term_of_imprisonment = meta["term_of_imprisonment"]     # dict

                criminals = meta["criminals"]           # list
                criminal = criminals[0]     # 数据中均只含一个被告

                money = meta["punish_of_money"]         # int
                accusations = meta["accusation"]        # list
                articles = meta["relevant_articles"]    # list

                # content of term_of_imprisonment
                death = term_of_imprisonment["death_penalty"]       # bool
                life_imprisonment = term_of_imprisonment["life_imprisonment"]   # bool
                imprisonment = term_of_imprisonment["imprisonment"]     # int

                if death:
                    imprisonment = DEATH_IMPRISONMENT
                elif life_imprisonment:
                    imprisonment = LIFE_IMPRISONMENT

                accusations_str = list2str_unicode_version(accusations)
                articles_str = list2str_unicode_version(articles)

                new_line = DATA_FORMAT.format(fact, criminal, money, accusations_str, articles_str, imprisonment)
                fcsv.write(new_line + BR)

                line = fjson.readline()

        fcsv.close()
        if DEBUG:
            print(".csv file dumped.")


    def run(self):
        for fname in os.listdir(self.data_dir):
            if ".json" in fname:
                fname = os.path.join(self.data_dir, fname)

                self.json2csv(fname)

        # self.tfidf = self.vectorizer.fit_transform(self.corpus)
        # print("shape: {}".format(self.tfidf.shape))

        # words = self.vectorizer.get_feature_names()
        # for i in xrange(len(self.corpus)):
        #     print("--------Document {}----------".format(i))
        #     for j in xrange(len(words)):
        #         if self.tfidf[i, j] > 1e-3:
        #             print(u"{}, {}".format(words[j], self.tfidf[i, j]))


    def handle(self, line):
        # 去掉日期
        line = re.sub("\d*年\d*月\d*日", "", line)

        # using thulac
        text = self.thulac.cut(line,
                               text=True)   # 返回分词文本，否则返回二维数组

        # 修正局部分词结果

        # 将「王」和「某某」合并为「王某某」
        text = re.sub(" 某某", u"某某", text)

        # 将「2000」和「元」合并为「2000元」
        text = re.sub(" 元", u"元", text)
        text = re.sub(" 余元", u"元", text)


        text = re.sub("价 格", "价格", text)

        words = text.split(" ")

        # remove the stopwords
        res = []
        for word in words:
            if word not in self.stopwords_set:
                res.append(word)


        # self.corpus.append(text)

        return " ".join(res)

if __name__ == "__main__":
    # fdir = os.path.join(DATA_DIR, "test/")
    handler = Preprocess(DATA_DIR, stopwords_fname="./stopwords.txt")
    handler.run()
