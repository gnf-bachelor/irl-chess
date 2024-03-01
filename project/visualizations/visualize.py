import os
import numpy as np
import pandas as pd

from os.path import join
from matplotlib import pyplot as plt


def plot_permuted_sunfish_weights(config_data, out_path, start_weight_idx=0, legend_names=['P', 'N', 'B', 'R', 'Q', 'K'], epoch=0):
    os.makedirs(out_path, exist_ok=True)
    start_plot_idx = config_data['permute_start_idx']
    end_plot_idx = config_data['permute_end_idx']
    save_every = config_data['save_every']
    epochs = config_data['epochs']

    weights = []
    for i in range(start_weight_idx, epochs, save_every):
        path = os.path.join(out_path, f'{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=None)
            weights.append(df.values.flatten())
        else:
            break
    weights = np.array(weights)
    X = np.repeat(np.arange(start_weight_idx, weights.shape[0], save_every), end_plot_idx - start_plot_idx).reshape((-1, end_plot_idx - start_plot_idx))

    plt.plot(X, np.array(weights)[:, start_plot_idx:end_plot_idx])
    plt.title('Sunfish weights over time')
    plt.xlabel('Epochs')
    plt.ylabel('Weight values')
    plt.legend(legend_names[start_plot_idx:end_plot_idx])
    plt.savefig(join(out_path, f'weights_at{epoch}.png'))
    plt.show()
