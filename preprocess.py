import pandas as pd
from os.path import *
import os
import cv2
import numpy as np
import glob


def combine():
    for subset in ['train', 'test']:
        print(subset)
        particles = glob.glob('data/%s/*_particles.npy' % subset)
        labels = glob.glob('data/%s/*_labels.npy' % subset)

        particles = [np.load(x) for x in particles]
        labels = [np.load(x) for x in labels]

        particles = np.concatenate(particles, axis=0)
        labels = np.concatenate(labels, axis=0)

        print('\tparticles:', particles.shape)
        print('\tlabels:', labels.shape)

        np.save('data/particles_%s.npy' % subset, particles)
        np.save('data/labels_%s.npy' % subset, labels)


if __name__ == '__main__':
    paths = glob.glob('data/train/*')

    for path in paths:
        print(basename(path))

        # directory to save particles
        if not exists(join(path, 'particles')):
            os.makedirs(join(path, 'particles'))

        # read in csv
        df = pd.read_csv(join(path, basename(path) + '.csv'))

        # maximum particle dimensions
        # maxdim = int(max(df['X_width'].max(), df['Y_height'].max()))
        maxdim = 256

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
        out = (out - out.mean()) / out.std()

        # organanize labels
        labels = np.array(labels)

        # # add blanks
        # blanks = np.random.normal(out.mean(), out.std(), size=out.shape)
        # blank_labels = np.zeros_like(labels)

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
