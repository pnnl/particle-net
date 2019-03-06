import pandas as pd
from os.path import *
import os
import cv2
import numpy as np

if __name__ == '__main__':
    # path to data
    path = '26B_EDX_Stg4_SEM'

    # directory to save particles
    if not exists(join(path, 'particles')):
        os.makedirs(join(path, 'particles'))

    # read in csv
    df = pd.read_csv('26b_edx_stg4_main.csv')

    # maximum particle dimensions
    maxdim = int(max(df['X_width'].max(), df['Y_height'].max()))

    # desired labels
    columns = ['C K', 'N K', 'O K', 'AlK', 'SiK', 'FeK', 'P K', 'K K', 'Area', 'Shape']

    # output containers
    output = []
    labels = []

    # iterate dataframe
    for i, row in df.iterrows():
        # field identifier
        fld = float(str(row['Field#'])[1:])

        # image path
        f = join(path, 'fld%04d' % fld, 'search.tif')

        if exists(f):
            # read image
            im = cv2.imread(f, 0)

            # particle location
            x0 = int(row['X_left'])
            w = int(row['X_width'])
            y0 = int(row['Y_low'])
            h = int(row['Y_height'])

            # invert y coordinate for opencv
            y0 = im.shape[0] - y0

            # make sure dims are even
            if h % 2 > 0:
                h += 1
                y0 = min(y0 + 1, im.shape[0])
            if w % 2 > 0:
                w += 1

            # crop
            cropped = im[y0 - h:y0, x0:x0 + w]

            # pad to maxdim
            padded = np.zeros((maxdim, maxdim))
            c = maxdim // 2

            padded[c - (h // 2): c + (h // 2), c - (w // 2):c + (w // 2)] = cropped

            # write image
            cv2.imwrite(join(path, 'particles', 'particle_fld%04d_%03d.png' % (fld, i)), padded)

            # append to array
            output.append(padded)
            labels.append(row[columns].values)

    # stack along first dim
    output = np.stack(output, axis=0)

    # normalize
    output = (output - output.mean()) / output.std()

    # organanize labels
    labels = np.array(labels)

    # add blanks
    blanks = np.random.normal(output.mean(), output.std(), size=output.shape)
    blank_labels = np.zeros_like(labels)

    # concat
    output = np.concatenate((output, blanks), axis=0)
    labels = np.concatenate((labels, blank_labels), axis=0)

    # shuffle
    idx = np.arange(output.shape[0])
    np.random.shuffle(idx)
    output = output[idx]
    labels = labels[idx]

    # save
    np.save('particles.npy', output)
    np.save('labels.npy', labels)
