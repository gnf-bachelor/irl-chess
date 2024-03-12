import os
import numpy as np
import pandas as pd

from os.path import join
from matplotlib import pyplot as plt

def char_to_idxs(plot_char: list[str]):
    char_to_idxs = {"P" : 0, "N" : 1, "B" : 2, "R" : 3, "Q" : 4, "K" : 5}
    return [char_to_idxs[char] for char in plot_char]

def idxs_to_char(idx_list: list[int]):
    idx_to_char = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
    return [idx_to_char[idx] for idx in idx_list]

def plot_permuted_sunfish_weights(config_data, out_path, start_weight_idx=0, legend_names=['P', 'N', 'B', 'R', 'Q', 'K'], epoch=None, **kwargs):
    accuracies = kwargs['accuracies'] if 'accuracies' in kwargs else None

    plot_char = char_to_idxs(config_data['plot_char'])
    save_every = config_data['save_every']
    epochs = config_data['epochs']
    R_true = np.array(config_data.get('R_true', [100, 280, 320, 479, 929, 60000]))

    plot_path = os.path.join(out_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    weights = []
    for i in range(start_weight_idx, epochs, save_every):
        path = os.path.join(out_path, f'{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=None)
            weights.append(df.values.flatten())
        else:
            if epoch is None:
                epoch = i
            break
    weights = np.array(weights)
    X = np.repeat(np.arange(start_weight_idx, weights.shape[0], save_every), len(plot_char)).reshape((-1, len(plot_char)))

    plt.plot(X, np.array(weights)[:, plot_char])
    plt.hlines(R_true[plot_char], 0, epoch, linestyles='--')
    plt.title('Sunfish weights over time')
    plt.xlabel('Epochs')
    plt.ylabel('Weight values')
    plt.legend([legend_names[idx] for idx in plot_char])
    plt.savefig(join(plot_path, f'weights_{epoch}.png'))
    plt.show()
    plt.cla()

    if accuracies is not None:
        accuracies = np.array(accuracies)
        plt.plot(np.arange(len(accuracies)), accuracies[:, 0])
        plt.plot(np.arange(len(accuracies)), accuracies[:, 1])
        plt.title('Accuracies over time')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Accuracy', 'Accuracy Prev'])
        plt.savefig(join(plot_path, f'accuracies_{epoch}.png'))
        plt.show()
        plt.cla()