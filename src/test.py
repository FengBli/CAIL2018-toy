#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime as dt

start = dt.now()

import json
import os
from utils import util
import h5py
import multiprocessing

from predictor import Predictor

data_path = "../data/CAIL2018-small-data/test_data"  # The directory of the input data
output_path = "../data/CAIL2018-small-data/test_result"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if not (x is None):
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if not (result["imprisonment"] is None):
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if not (x is None):
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex


if __name__ == "__main__":
    user = Predictor()
    cnt = 0

    point1 = dt.now()

    print('load model time:', (point1 - start).seconds, 's.')


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict(fact)

        for a in range(0, len(result)):
            result[a] = format_result(result[a])

        return result


    if util.DEBUG:
        print("start predict...")

    for file_name in os.listdir(data_path):
        inf = open(os.path.join(data_path, file_name), "r", encoding='utf-8')
        ouf = open(os.path.join(output_path, file_name), "w", encoding='utf-8')

        fact = []
        nums = 0
        tm0 = dt.now()
        for line in inf:
            fact.append(json.loads(line)["fact"])
            nums += 1
            if nums % 2000 == 0:
                print("solved:", nums, " cost time:", (dt.now() - tm0).seconds, "s.")
            if len(fact) == get_batch():
                result = solve(fact)
                cnt += len(result)
                for x in result:
                    print(json.dumps(x), file=ouf)
                fact = []

        if len(fact) != 0:
            result = solve(fact)
            cnt += len(result)
            for x in result:
                print(json.dumps(x), file=ouf)
            fact = []

        inf.close()
        ouf.close()

    point2 = dt.now()

    print("total time:", (point2 - point1).seconds, 's.')

    if util.DEBUG:
        print("DEBUG: prediction work finished.")
