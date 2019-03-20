import keras
import numpy as np
from train import test_train_split
from os.path import *
import pandas as pd


columns = ['C K', 'N K', 'O K', 'AlK', 'SiK', 'FeK', 'P K', 'K K', 'Area', 'Shape']


def evaluate(network, data, labels, test_size=0.33):
    # load data
    x = np.load(data)
    y = np.load(labels)

    # reshape
    n, m, k = x.shape
    x = x.reshape((n, m, k, 1))

    # test/train split
    mask = test_train_split(x, test_size=test_size)
    x_train = x[mask]
    x_validation = x[~mask]

    y_train = y[mask]
    y_validation = y[~mask]

    # create model
    model = keras.models.load_model(network)

    z_train = model.predict(x_train)
    z_validation = model.predict(x_validation)

    true_train = pd.DataFrame(data=y_train, columns=columns)
    true_validation = pd.DataFrame(data=y_validation, columns=columns)

    pred_train = pd.DataFrame(data=z_train, columns=columns)
    pred_validation = pd.DataFrame(data=z_validation, columns=columns)

    true_train.to_csv('result/true_train.tsv', sep='\t', index=False)
    true_validation.to_csv('result/true_test.tsv', sep='\t', index=False)
    pred_train.to_csv('result/predicted_train.tsv', sep='\t', index=False)
    pred_validation.to_csv('result/predicted_test.tsv', sep='\t', index=False)


if __name__ == '__main__':
    np.random.seed(777)
    evaluate('result/convnet.h5', 'data/particles.npy', 'data/labels.npy')