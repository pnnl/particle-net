import pandas as pd
from os.path import *
import os
import cv2
import numpy as np
import glob
# import matplotlib.pyplot as plt
# import seaborn as sns


def combine():
    # for subset in ['train', 'test']:
    # print(subset)
    particles = glob.glob('data/parts/*_particles.npy')
    labels = glob.glob('data/parts/*_labels.npy')

    data = [np.load(x) for x in labels]
    data = np.concatenate(data, axis=0)

    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    data = data[idx]

    print('\tlabels:', data.shape)
    np.save('data/labels.npy', data)

    data = [np.load(x) for x in particles]
    data = np.concatenate(data, axis=0)
    data = data[idx]

    print('\tparticles:', data.shape)
    np.save('data/particles.npy', data)


def process():
    paths = glob.glob('data/test/*/')
    paths = [dirname(x) for x in paths]

    # dims = []
    for path in paths:
        print(basename(path))

        # directory to save particles
        if not exists(join(path, 'particles')):
            os.makedirs(join(path, 'particles'))

        # read in csv
        df = pd.read_csv(join(path, basename(path) + '.csv'))

        # maximum particle dimensions
        # maxdim = np.max(np.column_stack([df['X_width'].values, df['Y_height'].values]), axis=-1)
        # dims.append(maxdim)
        maxdim = 96

        # desired labels
        columns = ['C K', 'N K', 'O K', 'AlK', 'SiK', 'FeK', 'P K', 'K K', 'Area', 'Shape']

        # out containers
        out = []
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

                if h > maxdim or w > maxdim:
                    pass
                else:
                    # crop
                    cropped = im[y0 - h:y0, x0:x0 + w]

                    # pad to maxdim
                    padded = np.zeros((maxdim, maxdim))
                    c = maxdim // 2

                    padded[c - (h // 2): c + (h // 2), c - (w // 2):c + (w // 2)] = cropped

                    # write image
                    cv2.imwrite(join(path, 'particles', 'particle_fld%04d_%03d.png' % (fld, i)), padded)

                    # append to array
                    out.append(padded)
                    labels.append(row[columns].values)

        # stack along first dim
        out = np.stack(out, axis=0)

        # normalize
        # out = (out - out.mean()) / out.std()

        # organanize labels
        labels = np.array(labels)

        # # add 10% blanks
        # blanks = np.random.normal(out.mean(), out.std(), size=(int(0.1 * out.shape[0]), maxdim, maxdim))
        # blank_labels = np.zeros((int(0.1 * out.shape[0]), maxdim, maxdim))

        # # concat
        # out = np.concatenate((out, blanks), axis=0)
        # labels = np.concatenate((labels, blank_labels), axis=0)

        # shuffle
        idx = np.arange(out.shape[0])
        np.random.shuffle(idx)
        out = out[idx]
        labels = labels[idx]

        # save
        np.save('data/%s_particles.npy' % basename(path), out)
        np.save('data/%s_labels.npy' % basename(path), labels)

    # dims = np.concatenate(dims, axis=0)
    # q = [50, 60, 70, 80, 90, 95, 98, 99, 99.5, 99.9]
    # p = np.percentile(dims, q)

    # for i, j in zip(q, p):
    #     print('%.1f%%: %s' % (i, j))

    # sns.distplot(dims, norm_hist=True, kde=False)
    # plt.axvline(x=p[-4], label='98th percentile', linestyle='--', c='k', linewidth=2)

    # plt.ylabel('density', fontweight='bold')
    # plt.xlabel('max dimension (pixels)', fontweight='bold')

    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    combine()
