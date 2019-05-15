import keras
import numpy as np
from train import test_train_split
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
    x_val = x[~mask]

    y_train = y[mask]
    y_val = y[~mask]

    # create model
    model = keras.models.load_model(network)

    z_train = model.predict(x_train)
    z_val = model.predict(x_val)

    true_train = pd.DataFrame(data=y_train, columns=columns)
    true_val = pd.DataFrame(data=y_val, columns=columns)

    pred_train = pd.DataFrame(data=z_train, columns=columns)
    pred_val = pd.DataFrame(data=z_val, columns=columns)

    true_train.to_csv('result/true_train.tsv', sep='\t', index=False)
    true_val.to_csv('result/true_test.tsv', sep='\t', index=False)
    pred_train.to_csv('result/predicted_train.tsv', sep='\t', index=False)
    pred_val.to_csv('result/predicted_test.tsv', sep='\t', index=False)

    train_error = 100 * np.mean(np.abs(y_train - z_train) / np.ma.masked_where(y_train == 0, np.abs(y_train)), axis=0)
    val_error = 100 * np.mean(np.abs(y_val - z_val) / np.ma.masked_where(y_val == 0, np.abs(y_val)), axis=0)

    print('\ntrain error:')
    print('\n'.join('{}: {:.1f}%'.format(*k) for k in zip(columns, train_error)))
    print('\nval error:')
    print('\n'.join('{}: {:.1f}%'.format(*k) for k in zip(columns, val_error)))


if __name__ == '__main__':
    np.random.seed(777)
    evaluate('result/convnet.h5', 'data/particles.npy', 'data/labels.npy')
