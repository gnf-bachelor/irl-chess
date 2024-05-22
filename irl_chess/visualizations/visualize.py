import os
import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join
from matplotlib import pyplot as plt

sunfish_palette_name = 'mako'
maia_palette_name = 'flare'

def char_to_idxs(plot_char: list[str]):
    char_to_idxs = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    return [char_to_idxs[char] for char in plot_char]


def idxs_to_char(idx_list: list[int]):
    idx_to_char = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
    return [idx_to_char[idx] for idx in idx_list]


def plot_BO_2d(opt, R_true, target_idxs, plot_idxs=None, plot_path=None, epoch=None):
    """

    :param opt:
    :param R_true:
    :param target_idxs:
    :return:
    """
    piece_names = ['PNBRQK'[idx] for idx in (plot_idxs if plot_idxs is not None else target_idxs)]
    piece_names_str = '_'.join(piece_names)

    plot_path_pieces = join(plot_path, piece_names_str)
    os.makedirs(plot_path_pieces, exist_ok=True)

    if len(target_idxs) > 2:
        if plot_idxs is not None:
            assert len(plot_idxs) == 2, 'there can only be 2 plot_idxs'
            plot_idxs = np.array(plot_idxs) - 1     # Pawn is never included
            # Now add the rest at the end of the array in order to use it for indexing:
            plot_idxs = np.array((*plot_idxs, *[i for i in range(len(target_idxs)) if i not in plot_idxs]))
        else:
            print('Only the first two target indexes are plotted!')
    n_grid = 1000
    linspace = np.linspace(0, opt.domain[0]['domain'][-1] + 10, n_grid).reshape((-1, 1))

    pgrid = np.array(np.meshgrid(linspace, linspace, indexing='ij'))
    # we then unfold the 4D array and simply pass it to the acqusition function
    pgrid_inp = np.concatenate((pgrid.reshape(2, -1).T, *[np.array([opt.X[np.argmin(opt.Y)][idx]]).repeat(n_grid ** 2).reshape((-1, 1)) for idx in plot_idxs[2:]],), axis=-1)
    # Rearrange so it can be used to reorder the columns
    pgrid_inp = pgrid_inp[:, plot_idxs] if plot_idxs is not None else pgrid_inp
    acq_img = opt.acquisition.acquisition_function(pgrid_inp)
    acq_img = (-acq_img - np.min(-acq_img)) / (np.max(-acq_img - np.min(-acq_img)))
    acq_img = acq_img.reshape(pgrid[0].shape[:2])
    mod_img = -opt.model.predict(pgrid_inp)[0]
    mod_img = mod_img.reshape(pgrid[0].shape[:2])

    accs = -opt.Y.reshape(-1)
    top_acc = np.maximum.accumulate(accs)
    plt.plot(top_acc)
    plt.title('Top accuracies over time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    # save
    if plot_path is not None and epoch is not None:
        plt.savefig(join(plot_path_pieces, f'accuracy_{epoch}.png'))
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(acq_img.T, origin='lower')
    ax1.set_xlabel(piece_names[0])
    ax1.set_ylabel(piece_names[1])
    ax1.set_title(f'Acquisition function after {epoch} iterations')
    ax2.imshow(mod_img.T, origin='lower')
    ax2.set_xlabel(piece_names[0])
    ax2.set_ylabel(piece_names[1])
    ax2.set_title(f'Model after {epoch} iterations')
    p1_true, p2_true = R_true[plot_idxs[:2] + 1]    # Here the pawn is included
    ax2.vlines([p1_true], 0, p2_true, color='red', linestyles='--')
    ax2.hlines([p2_true], 0, p1_true, color='red', linestyles='--')
    ax2.scatter(*opt.X[:, plot_idxs[:2]].T, color='red', marker='x', )
    # save
    if plot_path is not None and epoch is not None:
        plt.savefig(join(plot_path_pieces, f'acquisition_{epoch}.png'))
    plt.tight_layout()
    plt.show()
    plt.cla()

def plot_R_BO(opt, R_true, target_idxs, epoch=None, save_path=False):
    target_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x = np.array([R_true[:-1]] * len(opt.Y))
    x[:, target_idxs] = opt.X
    c = np.hstack((x, -opt.Y))
    cumulative_argmax = np.array([c[np.argmax(c[:i + 1, -1])] for i in range(len(c))])
    for i, values in enumerate(cumulative_argmax[:, :-1].T):
        plt.plot(values, c=target_colors[i])
    plt.hlines(R_true[:-1],0, c.shape[0]-1, colors=target_colors, linestyle='--')
    plt.suptitle('Bayesian Optimisation')
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'), loc='lower right')
    if save_path:
        plt.savefig(os.path.join(save_path, f'weights_over_time_{epoch}.png'))
    plt.show()


def plot_R_weights(config_data, out_path, start_weight_idx=0, legend_names=['P', 'N', 'B', 'R', 'Q', 'K'], epoch=None,
                   **kwargs):
    accuracies = kwargs['accuracies'] if 'accuracies' in kwargs else None
    bayesian_args = kwargs['bayesian_args'] if 'bayesian_args' in kwargs else None
    filename_addition = kwargs['filename_addition'] if 'filename_addition' in kwargs else ''
    plot_char = char_to_idxs(config_data['plot_char'])
    R_true = np.array(config_data.get('R_true', [100, 280, 320, 479, 929, 60000]))

    plot_path = os.path.join(out_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    weights = []
    for i in range(start_weight_idx, epoch+1, ):
        path = os.path.join(out_path, f'weights/{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=None)
            weights.append(df.values.flatten())
        else:
            if epoch is None:
                epoch = i
            break
    weights = np.array(weights)
    # assert weights.shape[0] == epoch+1, f"Error: weights.shape[0]: {weights.shape[0]} is not equal to epoch: {epoch}"
    X = np.repeat(np.arange(0, weights.shape[0], 1) + start_weight_idx,
                  len(plot_char)).reshape((-1, len(plot_char)))
    colors = sns.color_palette(sunfish_palette_name, n_colors=len(plot_char))
    for x, y, color, plot_char in zip(X.T, weights[:, plot_char].T, colors, plot_char):
        plt.plot(x, y, c=color, label=legend_names[plot_char])
        plt.hlines(R_true[plot_char], 0, x[-1], colors=color, linestyles='--')
    plt.title(f'Sunfish weights over time for ELO {config_data["min_elo"]}-{config_data["max_elo"]} on '
              f'{"player" if "player" in config_data["move_function"] else "sunfish"} moves')
    plt.xlabel('Epochs')
    plt.ylabel('Weight values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig(join(plot_path, f'weights_{epoch}{filename_addition}.svg'))
    plt.show()
    plt.close()

    if accuracies is not None:
        accuracies = np.array(accuracies)
        plt.plot(np.arange(len(accuracies)), accuracies[:, 0])
        plt.plot(np.arange(len(accuracies)), accuracies[:, 1])
        plt.title('Accuracies over time')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(['Accuracy', 'Accuracy Prev'])
        plt.savefig(join(plot_path, f'accuracies_{epoch}.svg'))
        plt.show()
        plt.cla()

    if bayesian_args is not None:
        plot_BO_2d(*bayesian_args, epoch=epoch, plot_path=plot_path)
