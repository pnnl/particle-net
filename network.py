from keras.layers import Input, Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam


class ConvNet(object):
    def __init__(self, dim=256, kernels=[7, 3, 3], filters=[64, 192, 480], maxpool=[2, 2, 2],
                 stride=[2, 2, 2], dropout=0.2, nlabels=2):
        self.dim = dim
        self.kernels = kernels
        self.filters = filters
        self.maxpool = maxpool
        self.stride = stride
        self.dropout = dropout
        self.nlabels = nlabels

    def create(self):
        # define input, embed
        x = Input(shape=(self.dim, self.dim, 1))

        # build filters
        for i, (f, k, m, s) in enumerate(zip(self.filters, self.kernels, self.maxpool, self.stride)):
            if i < 1:
                h = Conv2D(f, (k, k), strides=s, activation='relu', padding='same', data_format='channels_last')(x)
            else:
                h = Conv2D(f, (k, k), strides=s, activation='relu', padding='same', data_format='channels_last')(h)

            h = MaxPooling2D(pool_size=(m, m), strides=s, data_format='channels_last')(h)

        # flatten
        h = Flatten()(h)

        # dropout
        h = Dropout(self.dropout)(h)

        # dense
        h = Dense(self.nlabels, activation='relu')(h)

        # output
        output = Dense(self.nlabels, activation='linear')(h)

        # define model
        self.network = Model(inputs=x,
                             outputs=output,
                             name='convnet')

        # optimizer
        opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1E-8, amsgrad=True)

        # compile autoencoder
        self.network.compile(optimizer=opt,
                             loss='mae',
                             metrics=['accuracy'])
