import gc
import json
import math
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from build_vocab import WordVocab
from pretrain_trfm import TrfmSeq2seq
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from transformers import T5EncoderModel, T5Tokenizer
from utils import split


def Kcat_predict(Ifeature, Label):
    for i in range(5):
        model = ExtraTreesRegressor()
        model.fit(Ifeature, Label)
        with open("PreKcat_new/" + str(i) + "_model.pkl", "wb") as f:
            pickle.dump(model, f)


if __name__ == "__main__":
    with open("Kcat_combination_0918_wildtype_mutant.json", "r") as file:
        datasets = json.load(file)
    # print(len(datasets))
    Label = [float(data["Value"]) for data in datasets]
    Smiles = [data["Smiles"] for data in datasets]
    for i in range(len(Label)):
        if Label[i] == 0:
            Label[i] = -10000000000
        else:
            Label[i] = math.log(Label[i], 10)
    with open("PreKcat_new/features_16838_PreKcat.pkl", "rb") as f:
        feature = pickle.load(f)
    Label = np.array(Label)
    Label_new = []
    feature_new = []
    for i in range(len(Label)):
        if -10000000000 < Label[i] and "." not in Smiles[i]:
            Label_new.append(Label[i])
            feature_new.append(feature[i])
    print(len(Label_new))
    Label_new = np.array(Label_new)
    feature_new = np.array(feature_new)
    Kcat_predict(feature_new, Label_new)
