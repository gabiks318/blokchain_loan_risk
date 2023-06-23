import pickle
from utils import etl
import svm_model
import resnet_model
import utils

if __name__ == '__main__':
    with open('data/HistoryPoolBorrowersStats.pkl', 'rb') as f:
        data = pickle.load(f)
    df = etl(data)
    # train_data = df.loc[:2]
    # svm = svm_model.get_trained_model(df.drop(columns=['label']), df['label'])
    # print(svm.predict(df.loc[:2].drop(columns=['label'])))
    # resnet = resnet_model.get_trained_model(df.drop(columns=['label']), df['label'])
    # print(resnet.predict(df.loc[:2].drop(columns=['label'])))
