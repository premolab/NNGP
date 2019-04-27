import numpy as np
from matplotlib import pyplot as plt

def plot_hist(ys, ax=None, figsize=(8, 6)):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    for y1, y2, name in ys:
        ax.hist(y1, bins=20, alpha=0.2, label=name+'-true')
        ax.hist(y2, bins=20, alpha=0.2, label=name+'-pred')
    ax.grid()
    ax.legend()
    plt.show()
    
def plot_region(Xi, Xj, Y, X=None, ax=None, figsize=(8, 6)):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
    cp = plt.contourf(Xi, Xj, Y)
    plt.colorbar(cp)
    if X is not None:
        plt.scatter(X.T[0], X.T[1], s=50, edgecolors='k', facecolors='w')
    plt.xlim([Xi.min(), Xi.max()])
    plt.ylim([Xi.min(), Xi.max()])
    plt.show()
    
    
def plot_history(logs, ax=None, figsize=(18, 6)):
    if ax is None:
        f, ax = plt.subplots(1, 3, figsize=(18, 6))
        
    n_epochs = logs['epochs']
    gap = logs['check_every']
    epoch_linspace = np.arange(0, n_epochs, gap)
    names = ['RMSE', 'MAE', 'MaxAE']

    for i in range(3):
        if len(np.array(logs['val']).shape) == 2:
            ax[i].plot(epoch_linspace, np.array(logs['train'])[:, i], label='Train')
        elif len(np.array(logs['val']).shape) == 3:
            mean = np.array(logs['train'])[:, :, i].mean(axis=1)
            std = np.array(logs['train'])[:, :, i].std(axis=1)
            ax[i].plot(epoch_linspace, mean, label='Train')
            ax[i].fill_between(epoch_linspace, mean-std, mean+std, alpha=0.2)
        if len(np.array(logs['val']).shape) == 2:
            ax[i].plot(epoch_linspace, np.array(logs['val'])[:, i], label='Val')
        elif len(np.array(logs['val']).shape) == 3:
            mean = np.array(logs['val'])[:, :, i].mean(axis=1)
            std = np.array(logs['val'])[:, :, i].std(axis=1)
            ax[i].plot(epoch_linspace, mean, label='Val')
            ax[i].fill_between(epoch_linspace, mean-std, mean+std, alpha=0.2)
        ax[i].grid()
        ax[i].set_title(names[i])
        ax[i].legend()

    plt.show()