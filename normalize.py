import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def labels():
    data = {}
    normed = {}
    logged = {}
    normlog = {}
    lognorm = {}

    data['data'] = np.load('data/labels.npy')

    # clip < 0
    data['data'] = np.where(data['data'] > 0, data['data'], 0)

    data['mean'] = np.mean(data['data'], axis=0)
    data['std'] = np.std(data['data'], axis=0)
    data['min'] = np.min(data['data'], axis=0)

    # normalize
    normed['data'] = (data['data'] - data['mean']) / data['std']

    normed['min'] = np.min(normed['data'], axis=0)

    # log transform
    logged['data'] = np.log(data['data'] - data['min'] + 1)

    logged['mean'] = np.mean(logged['data'], axis=0)
    logged['std'] = np.std(logged['data'], axis=0)

    # normlog
    normlog['data'] = np.log(normed['data'] - normed['min'] + 1)

    # lognorm
    lognorm['data'] = (logged['data'] - logged['mean']) / logged['std']

    # print('normlog')
    # print(np.mean(normed['data'], axis=0))
    # print(np.std(normed['data'], axis=0))

    # print('\nlognorm')
    # print(np.mean(lognorm['data'], axis=0))
    # print(np.std(lognorm['data'], axis=0))

    np.save('data/labels_norm.npy', normed['data'])
    np.save('data/labels_log.npy', logged['data'])
    np.save('data/labels_normlog.npy', normlog['data'])
    np.save('data/labels_lognorm.npy', lognorm['data'])

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(9, 2), dpi=150)
    labels = ['original', 'normalized', 'logged', 'norm+log', 'log+norm']
    for i, subset in enumerate([data, normed, logged, normlog, lognorm]):
        ax[i].set_title(labels[i])
        for j in range(subset['data'].shape[1]):
            sns.kdeplot(subset['data'][:, j], ax=ax[i])

    plt.show()


def images():
    data = {}
    data['data'] = np.load('data/particles.npy')

    data['mean'] = np.mean(data['data'], axis=0)
    data['std'] = np.std(data['data'], axis=0)

    data['data'] = (data['data'] - data['mean']) / data['std']

    data['data'] = np.nan_to_num(data['data'])

    np.save('data/particles_norm.npy', data['data'])


if __name__ == '__main__':
    labels()
    images()
