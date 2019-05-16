import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def labels():
    data = {}
    normed = {}
    logged = {}
    normlog = {}
    lognorm = {}

    data['test'] = np.load('data/labels_test.npy')
    data['train'] = np.load('data/labels_train.npy')

    # clip < 0
    data['test'] = np.where(data['test'] > 0, data['test'], 0)
    data['train'] = np.where(data['train'] > 0, data['train'], 0)

    data['all'] = np.concatenate([data['test'], data['train']], axis=0)
    data['mean'] = np.mean(data['all'], axis=0)
    data['std'] = np.std(data['all'], axis=0)
    data['min'] = np.min(data['all'], axis=0)

    # normalize
    normed['test'] = (data['test'] - data['mean']) / data['std']
    normed['train'] = (data['train'] - data['mean']) / data['std']

    normed['all'] = np.concatenate([normed['test'], normed['train']], axis=0)
    normed['min'] = np.min(normed['all'], axis=0)

    # log transform
    logged['test'] = np.log(data['test'] - data['min'] + 1)
    logged['train'] = np.log(data['train'] - data['min'] + 1)

    logged['all'] = np.concatenate([logged['test'], logged['train']], axis=0)
    logged['mean'] = np.mean(logged['all'], axis=0)
    logged['std'] = np.std(logged['all'], axis=0)

    # normlog
    normlog['test'] = np.log(normed['test'] - normed['min'] + 1)
    normlog['train'] = np.log(normed['train'] - normed['min'] + 1)

    # lognorm
    lognorm['test'] = (logged['test'] - logged['mean']) / logged['std']
    lognorm['train'] = (logged['train'] - logged['mean']) / logged['std']

    print('normlog')
    print(np.mean(normed['test'], axis=0))
    print(np.std(normed['test'], axis=0))

    print('\nlognorm')
    print(np.mean(lognorm['test'], axis=0))
    print(np.std(lognorm['test'], axis=0))

    np.save('data/labels_norm_test.npy', normed['test'])
    np.save('data/labels_norm_test.npy', normed['train'])

    np.save('data/labels_log_test.npy', normed['test'])
    np.save('data/labels_log_train.npy', normed['train'])

    np.save('data/labels_normlog_test.npy', normlog['test'])
    np.save('data/labels_normlog_train.npy', normlog['train'])

    np.save('data/labels_lognorm_test.npy', lognorm['test'])
    np.save('data/labels_lognorm_train.npy', lognorm['train'])

    fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(9, 2), dpi=150)
    labels = ['original', 'normalized', 'logged', 'norm+log', 'log+norm']
    for i, subset in enumerate([data, normed, logged, normlog, lognorm]):
        ax[i].set_title(labels[i])
        for j in range(subset['train'].shape[1]):
            sns.kdeplot(subset['train'][:, j], ax=ax[i])

    plt.show()


def images():
    data = {}
    data['train'] = np.load('data/particles_train.npy')
    data['test'] = np.load('data/particles_test.npy')

    data['all'] = np.concatenate([data['train'], data['test']], axis=0)
    data['mean'] = np.mean(data['all'], axis=0)
    data['std'] = np.std(data['all'], axis=0)

    data['train'] = (data['train'] - data['mean']) / data['std']
    data['test'] = (data['test'] - data['mean']) / data['std']

    np.save('data/particles_norm_train.npy'. data['train'])
    np.save('data/particles_norm_test.npy', data['test'])


if __name__ == '__main__':
    # labels()
    # images()
