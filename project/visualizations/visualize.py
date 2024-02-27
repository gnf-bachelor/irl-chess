import os
import numpy as np
import pandas as pd

from os.path import join
from matplotlib import pyplot as plt


def plot_permuted_sunfish_weights(epochs, save_every, out_path, start_idx=0, ignore_idx=2, legend_names=['P', 'N', 'B', 'R', 'Q', 'K']):
    weights = []
    for i in range(start_idx, epochs, save_every):
        path = os.path.join(out_path, f'{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=None)
            weights.append(df.values.flatten())
        else:
            break
    weights = np.array(weights)
    X = np.repeat(np.arange(start_idx, weights.shape[0], save_every), 6 - ignore_idx).reshape((-1, 6 - ignore_idx))

    plt.plot(X, np.array(weights)[:, :6 - ignore_idx])
    plt.title('Sunfish weights over time')
    plt.xlabel('Epochs')
    plt.ylabel('Weight values')
    plt.legend(legend_names)
    plt.savefig(join(out_path, 'weights_over_time.png'))
    plt.show()
