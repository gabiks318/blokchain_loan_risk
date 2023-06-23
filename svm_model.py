import pandas as pd
from sklearn.svm import SVC


def get_trained_model(X: pd.DataFrame, y: pd.DataFrame):
    svm = SVC()
    svm.fit(X, y)
    return svm
