import keras
import numpy as np
from os.path import *
from network import ConvNet


def test_train_split(x, test_size=0.1):
    idx = np.random.choice(np.arange(len(x)), size=int(len(x) * test_size), replace=False)
    mask = np.ones(len(x)).astype('bool')
    mask[idx] = False
    return mask


def train(data, labels, output, test_size=0.33,
          batch_size=32, epochs=10000, patience=100):
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
    model = ConvNet(dim=k, kernels=[7, 3, 3], filters=[64, 192, 480], maxpool=[3, 3, 3],
                    stride=[2, 2, 2], dropout=0.2, nlabels=y.shape[-1])
    model.create()

    # early stop
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

    # checkpoint
    checkpoint = keras.callbacks.ModelCheckpoint(join(output, 'convnet.h5'), monitor='val_loss',
                                                 save_best_only=True, mode='min')

    # print
    print(model.network.summary())

    # train
    model.network.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(x_validation, y_validation),
                      callbacks=[early_stop, checkpoint],
                      shuffle=True,
                      verbose=2)


if __name__ == '__main__':
    np.random.seed(777)
    train('data/particles_train.npy', 'data/labels_lognorm_train.npy', 'result', epochs=1000, batch_size=128)
