import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = {}
normed = {}
logged = {}
normlog = {}

data['test'] = np.load('data/labels_test.npy')
data['train'] = np.load('data/labels_train.npy')

data['all'] = np.concatenate([data['test'], data['train']], axis=0)
data['mean'] = np.mean(data['all'], axis=0)
data['std'] = np.std(data['all'], axis=0)

# normalize
normed['test'] = (data['test'] - data['mean']) / data['std']
normed['train'] = (data['train'] - data['mean']) / data['std']

# log transform
logged['test'] = np.log(data['test'])
logged['train'] = np.log(data['train'])

# normlog
normlog['test'] = np.log(normed['test'])
normlog['train'] = np.log(normed['train'])

# np.save('data/labels_norm_test.npy', normed['test'])
# np.save('data/labels_norm_test.npy', normed['train'])

# np.save('data/labels_log_test.npy', normed['test'])
# np.save('data/labels_log_train.npy', normed['train'])

# np.save('data/labels_normlog_test.npy', normlog['test'])
# np.save('data/labels_normlog_train.npy', normlog['train'])

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(9, 2), dpi=150)
labels = ['original', 'normalized', 'logged', 'normalized+logged']
for i, subset in enumerate([data, normed, logged, normlog]):
    ax[i].set_title(labels[i])
    for j in range(subset['train'].shape[1]):
        sns.kdeplot(subset['train'][:, j], ax=ax[i])

plt.show()
