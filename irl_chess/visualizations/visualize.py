import os
import numpy as np
import pandas as pd
import seaborn as sns

from os.path import join
from matplotlib import pyplot as plt

sunfish_palette_name = 'mako'
maia_palette_name = 'flare'

result_strings = ['Result', 'RpstResult', 'RHResult']

def char_to_idxs(plot_char: list[str]):
    char_to_idxs = {"P": 0, "N": 1, "B": 2, "R": 3, "Q": 4, "K": 5}
    return [char_to_idxs[char] for char in plot_char]


def idxs_to_char(idx_list: list[int]):
    idx_to_char = {0: "P", 1: "N", 2: "B", 3: "R", 4: "Q", 5: "K"}
    return [idx_to_char[idx] for idx in idx_list]

def Hbool_to_idxs(plot_char: list[bool]):
    #char_to_idxs = {"PA": 0, "KS": 1, "PS": 2}
    Hchar = ["PA", "KS", "PS"]
    return list(np.arange(len(Hchar))[plot_char])


def idxs_to_Hchar(idx_list: list[int]):
    idx_to_char = {0: "PA", 1: "KS", 2: "PS"}
    return [idx_to_char[idx] for idx in idx_list]

def load_weights(out_path, result: str, start_weight_idx=0, epoch=None): # Load all weights up to epoch
    weights = []
    for i in range(start_weight_idx, epoch+1, ):
        weights_i = load_weights_epoch(out_path, result, epoch = i)
        if weights_i is not None:
            weights.append(weights_i)
    return np.array(weights)

def load_weights_epoch(out_path, result: str, epoch):
    path = os.path.join(out_path, f'weights/{epoch}.csv')
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=None)
        return df[result].dropna().values.flatten()
    return None


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
    show = kwargs.get('show', False)
    close_when_done = kwargs.get('close_when_done', True)
    save_path = kwargs.get('save_path', None)

    plot_path = os.path.join(out_path, 'plots')
    os.makedirs(plot_path, exist_ok=True)

    plot_char = char_to_idxs(config_data['plot_char'])
    if plot_char:
        RP_plot_path = os.path.join(plot_path, 'RP')
        os.makedirs(RP_plot_path, exist_ok=True)
        weights = load_weights(out_path, 'Result', start_weight_idx=start_weight_idx, epoch=epoch)
        RP_true = np.array(config_data.get('RP_true', [100, 280, 320, 479, 929, 60000]))
        plot_weights(weights, RP_true, start_weight_idx, plot_char, legend_names, config_data, RP_plot_path, epoch+1,
                     filename_addition= 'RP'+filename_addition, show=show, close_when_done=close_when_done, save_path=save_path)

    plot_pst_char = char_to_idxs(config_data['plot_pst_char'])
    if plot_pst_char:
        Rpst_plot_path = os.path.join(plot_path, 'Rpst')
        os.makedirs(Rpst_plot_path, exist_ok=True)
        weights = load_weights(out_path, 'RpstResult', start_weight_idx=start_weight_idx, epoch=epoch)
        Rpst_true = np.array(config_data.get('Rpst_true', [1, 1, 1, 1, 1, 1]))
        plot_weights(weights, Rpst_true, start_weight_idx, plot_pst_char, legend_names, config_data, Rpst_plot_path, epoch+1,
                     filename_addition= 'Rpst'+filename_addition, show=show, save_path=save_path)

    plot_H_char = Hbool_to_idxs(config_data['plot_H'])
    if plot_H_char:
        RH_plot_path = os.path.join(plot_path, 'RH')
        os.makedirs(RH_plot_path, exist_ok=True)
        weights = load_weights(out_path, 'RHResult', start_weight_idx=start_weight_idx, epoch=epoch)
        RH_true = np.array(config_data.get('RH_true', [0, 0, 0])) # Perhaps delete this, as there is no ground truth. 
        plot_weights(weights, RH_true, start_weight_idx, plot_H_char, ['PA','KS','PS'], config_data, RH_plot_path, epoch+1,
                     filename_addition= 'RH'+filename_addition, show=show, save_path=save_path)
        

    
def plot_weights(weights, weights_true, start_weight_idx, plot_char, legend_names, config_data, plot_path, epoch, filename_addition, show=False, close_when_done=True, save_path=None):
    X = np.repeat(np.arange(0, weights.shape[0], 1) + start_weight_idx,
                  len(plot_char)).reshape((-1, len(plot_char)))
    colors = sns.color_palette(sunfish_palette_name, n_colors=len(plot_char))
    alpha = config_data.get('alpha', 1)
    for x, y, color, plot_char in zip(X.T, weights[:, plot_char].T, colors, plot_char):
        plt.plot(x, y, c=color, label=legend_names[plot_char] if config_data.get('add_legend', True) else None, alpha=alpha)
        if len(weights_true): plt.hlines(weights_true[plot_char], 0, x[-1], colors=color, linestyles='--')
    plt.title(f'Sunfish weights over time for ELO {config_data["min_elo"]}-{config_data["max_elo"]} on '
              f'{"player" if "player" in config_data["move_function"] else "sunfish"} moves') if not config_data.get('plot_title', False) else plt.title(config_data['plot_title'])
    plt.xlabel('Epochs')
    plt.ylabel('Weight values')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    os.makedirs(plot_path, exist_ok=True)
    plt.savefig(join(plot_path, f'weights_{epoch}{filename_addition}.svg')) if save_path is None else plt.savefig(save_path)
    if show: plt.show()
    if close_when_done: plt.close()


if __name__ == '__main__':
    from irl_chess import fix_cwd, load_config, load_model_functions, union_dicts, create_result_path
    fix_cwd()
    base_config_data, model_config_data = load_config()
    model, model_result_string = load_model_functions(base_config_data)
    config_data = union_dicts(base_config_data, model_config_data)
    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)
    assert os.path.isdir(out_path), 'The result path does not exist, which means that result data has not been generated for this configuration.\
          Run run_model with the correct configs to generate results and plots.'
    
    for epoch in range(config_data['epochs']): # Check how many epochs the weights have been saved for.
        path = os.path.join(out_path, f'weights/{epoch}.csv')
        if os.path.exists(path):
            epoch += 1
        else:
            epoch -= 1
            print("Weights have been saved for", epoch+1, "epochs.")
            break

    plot_R_weights(config_data, out_path, start_weight_idx=0, epoch=epoch, show=True)