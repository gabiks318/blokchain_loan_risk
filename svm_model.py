import pandas as pd
from sklearn.svm import SVR


def get_trained_model(X: pd.DataFrame, y: pd.DataFrame):
    svm = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svm.fit(X, y)
    return svm
