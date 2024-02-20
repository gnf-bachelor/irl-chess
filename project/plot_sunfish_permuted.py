import os
import json
from os.path import join

if __name__ == '__main__':
    if os.getcwd().split('\\')[-1] != 'irl-chess':
        os.chdir('../')
    from project import plot_permuted_sunfish_weights

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'), 'r') as file:
        config_data = json.load(file)

    n_files = config_data['n_files']
    min_elo = config_data['min_elo']
    max_elo = config_data['max_elo']
    delta = config_data['delta']
    n_boards = config_data['n_boards']
    search_depth = config_data['search_depth']
    epochs = config_data['epochs']
    save_every = config_data['save_every']
    permute_all = config_data['permute_all']
    R_noisy_vals = config_data['R_noisy_vals']
    permute_end_idx = config_data['permute_end_idx']

    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = join(path_result, f'{permute_all}-{min_elo}-{max_elo}-{search_depth}-{n_boards}-{delta}'
                    + (f'-{R_noisy_vals}' if type(R_noisy_vals) is int else '') + f'-{permute_end_idx}')
    plot_permuted_sunfish_weights(epochs=epochs, save_every=save_every, out_path=out_path)
