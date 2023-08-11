import pickle

import pandas as pd

from analysis_plots import analyze_data
from utils import etl
import svm_model
import resnet_model
import utils


def process_data():
    datas = ["compound_v3_train.pkl", "compound_v3_test.pkl", "compound_v3_validation.pkl"]
    for data in datas:
        with open("data/" + data, 'rb') as f:
            df = etl(pickle.load(f))
        df.to_csv("data/" + data[:-4] + ".csv")


def train_model(model_name: str):
    train_data = pd.read_csv("data/compound_v3_train.csv")
    validation_data = pd.read_csv("data/compound_v3_validation.csv")
    if model_name == "svm":

        svm = svm_model.get_trained_model(train_data.drop(columns=['label', 'loan_repay_percentage']),
                                          train_data['label'])
        utils.evaluate_model(svm, train_data, validation_data)
    elif model_name == "resnet":
        resnet = resnet_model.get_trained_model(train_data.drop(columns=['label', 'loan_repay_percentage']),
                                                train_data['label'])
        utils.evaluate_model(resnet, train_data, validation_data)
    else:
        raise ValueError("Not supported model")


if __name__ == '__main__':
    train_data = pd.read_csv("data/compound_v3_train.csv", index_col=0)
    # process_data()
    # train_model("resnet")
    analyze_data(train_data)
