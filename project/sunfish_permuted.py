import json
import os
import pickle
from copy import copy
from shutil import copy2
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt


def run_sun(df,
            R_sunfish,
            R_noisy_vals=0,
            search_depth=3,
            min_elo=1000,
            max_elo=1200,
            n_boards=1000,
            delta=20.,
            permute_all=1,
            permute_end_idx=-1,
            sd_noise=50,
            epochs=1,
            depth=3,
            path_result=None,
            save_every=1000,
            ):
    """

    :param df:
    :param min_elo:
    :param max_elo:
    :param n_boards:
    :param sd_noise:
    :param depth:
    :return:
    """

    R_noisy = copy(R_sunfish)
    R_noisy[1:3] = R_noisy_vals  # Keep the pawn constant
    # R_noisy[1:] += np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape[0] - 1)

    boards, _ = get_midgame_boards(df, n_boards, min_elo=min_elo, max_elo=max_elo, sunfish=False)
    n_boards = len(boards)
    print(f'Found {n_boards} boards, setting n_boards to {n_boards}')

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = join(path_result, f'{permute_all}-{min_elo}-{max_elo}-{search_depth}-{n_boards}-{delta}-{R_noisy_vals}-{max(permute_end_idx, 0)}')
    os.makedirs(out_path, exist_ok=True)
    copy2(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'),
          join(os.path.dirname(out_path), 'config.json'))

    boards, moves_sunfish = get_sunfish_moves(boards=boards, depth=depth, out_path=out_path)
    R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=delta, epochs=epochs, save_every=save_every,
                     save_path=out_path, permute_all=permute_all, permute_end_idx=permute_end_idx)

    from project import plot_permuted_sunfish_weights
    plot_permuted_sunfish_weights(epochs=epochs, save_every=save_every, out_path=out_path)

    return R_


def get_sunfish_moves(boards, depth, out_path):
    """
    If the moves have already been calculated for the configuration
    this function just reads the file, otherwise the moves are found
    and saved as strings in the SAN format.
    :param depth:
    :param out_path:
    :return:
    """

    sunfish_moves_path = os.path.join(out_path, 'sunfish_moves.csv')
    if os.path.exists(sunfish_moves_path):
        print(f'Loaded saved SUNFISH moves from {sunfish_moves_path}')
        moves_sunfish = list(pd.read_csv(sunfish_moves_path, index_col=None, header=None).values.flatten())
    else:
        moves_sunfish = []
        for board in tqdm(boards, desc='Getting Sunfish moves'):
            move, Q = get_best_move(board, R_sunfish, depth=depth, san=True)
            moves_sunfish.append(move)
        pd.DataFrame(moves_sunfish).to_csv(sunfish_moves_path, index=None, header=None)

    return boards, moves_sunfish


if __name__ == '__main__':
    if os.getcwd().split('\\')[-1] != 'irl-chess':
        os.chdir('../')
    from project import policy_walk, policy_walk_multi, get_midgame_boards, get_best_move, piece, download_lichess_pgn

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
    overwrite = config_data['overwrite']
    permute_end_idx = config_data['permute_end_idx']

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    datapaths = download_lichess_pgn(websites_filepath, file_path_data, n_files=n_files, overwrite=overwrite)
    df = pd.read_csv(datapaths[0], index_col=None)
    for path in tqdm(datapaths[1:], desc='Contatenating DataFrames'):
        df_ = pd.read_csv(path, index_col=None)
        df = pd.concat((df, df_), axis=0)
    df.dropna(inplace=True)

    R_sunfish = np.array([val for val in piece.values()]).astype(float)
    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')

    result = run_sun(df,
                     R_noisy_vals=R_noisy_vals,
                     R_sunfish=R_sunfish,
                     min_elo=min_elo,
                     search_depth=search_depth,
                     max_elo=max_elo,
                     depth=search_depth,
                     permute_end_idx=permute_end_idx,
                     n_boards=n_boards,
                     permute_all=permute_all,
                     path_result=path_result,
                     save_every=save_every,
                     epochs=epochs,
                     delta=delta)
