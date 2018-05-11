#-*- coding: utf-8 -*-
"""
将json格式数据转化为csv格式，分隔符设置为"$$"。
另外，为了便于对刑期的预测，将**死刑**和**无期徒刑**数值化，分别令为600和800（单位：月）。
注：
- 此两数值后续根据情况进行调整
- 对数据中的 "[1, 2, 3]"这样的str，可以使用eval(str)得到对应list
"""

from __future__ import print_function
import codecs
import json
import os


LIFE_IMPRISONMENT = 600
DEATH_IMPRISONMENT = 800

BR = "\n"
TITLE = u"fact$$criminal$$money$$accusations$$articles$$imprisonment"
DATA_FORMAT = u"{}$${}$${}$${}$${}$${}"

def json2csv(fname):
    """
    Convert .json file to .csv file, using the same filename with ".csv" replacing ".json"

    param:
    -------
    fname:  the .json file to be converted
    """

    fcsv_name = fname.replace("json", "csv")

    fcsv = codecs.open(fcsv_name, "a+", encoding="utf-8")
    fcsv.write(TITLE+BR)

    with codecs.open(fname, "r", encoding="utf-8") as fjson:
        line = fjson.readline()
        while line:
            lineDict = json.loads(line, encoding="utf-8")

            fact = lineDict["fact"]     # string
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


def list2str_unicode_version(lst):
    """
    将list转换为unicode str，展示对应汉字而非unicode十六进制编码
    可以使用eval(str)得到对应list
    """
    assert len(lst) >= 1
    str = u"[{}".format(lst[0])
    if len(lst) > 1:
        for item in lst[1:]:
            str += u", {}".format(item)
    str += u"]"

    return str


def main():
    path = "./small-data"
    for fname in os.listdir(path):
        if ".json" in fname:
            fname = os.path.join(path, fname)
            print("handling {}...\t".format(fname),end="")
            json2csv(fname)
            print("Done")

if __name__ == '__main__':
    main()
