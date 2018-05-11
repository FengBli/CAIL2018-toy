# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs

f1 = codecs.open("data_train_1.json", "a+", encoding="utf-8")
f2 = codecs.open("data_train_2.json", "a+", encoding="utf-8")
f3 = codecs.open("data_train_3.json", "a+", encoding="utf-8")

with codecs.open("data_train.json", "r", encoding="utf-8") as f:
    line = f.readline()
    line_cnt = 0
    while line:
        line_cnt += 1

        if line_cnt <= 50000:
            f1.write(line)
        elif line_cnt <= 100000:
            f1.close()
            f2.write(line)
        else:
            f2.close()
            f3.write(line)
        line = f.readline()

    f3.close()
print("total line: {}".format(line_cnt))

