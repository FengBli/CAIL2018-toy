import util
import os
import json
from sklearn.metrics import f1_score
import pandas as pd


if __name__ == '__main__':
    data_path = os.path.join(util.DATA_DIR, "test/data_valid.json")
    output_path = os.path.join(util.DATA_DIR, "output/data_valid.json")

    truth = []
    with open(data_path, 'r') as f:
        line = f.readline()
        while line:
            article = json.loads(line.strip())["meta"]["relevant_articles"][0]
            truth.append(article)
            line = f.readline()

    predict = []
    with open(output_path, 'r') as f:
        line = f.readline()
        while line:
            article = json.loads(line.strip())["articles"][0]
            predict.append(article)
            line = f.readline()

    predict = pd.Series(predict)
    truth = pd.Series(truth)

    micro_f1 = f1_score(truth, predict, average='micro')
    macro_f1 = f1_score(truth, predict, average="macro")

    print("score = {}".format((micro_f1 + macro_f1) / 2.0 * 100))
