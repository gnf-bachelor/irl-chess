import json
import os
import pickle
from copy import copy
from shutil import copy2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from os.path import join
import pickle
import matplotlib.pyplot as plt
from project.chess_utils.utils import alpha_beta_search


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
            quiesce=True,
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
    R_noisy[1:permute_end_idx] = R_noisy_vals  # Keep the pawn constant
    # R_noisy[1:] += np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape[0] - 1)

    boards, _ = get_midgame_boards(df, n_boards, min_elo=min_elo, max_elo=max_elo, sunfish=False)
    n_boards = len(boards)
    print(f'Found {n_boards} boards, setting n_boards to {n_boards}')

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = join(path_result, f'{permute_all}-{min_elo}-{max_elo}-{search_depth}-{n_boards}-{delta}-{R_noisy_vals}-{max(permute_end_idx, 0)}-{quiesce}')
    os.makedirs(out_path, exist_ok=True)
    copy2(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'),
          join(out_path, 'config.json'))

    boards, moves_sunfish = get_sunfish_moves(R_sunfish=R_sunfish, boards=boards, depth=depth, out_path=out_path, overwrite=overwrite, quiesce=quiesce)
    R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=delta, epochs=epochs, save_every=save_every,
                     save_path=out_path, permute_all=permute_all, permute_end_idx=permute_end_idx, quiesce=quiesce)

    from project import plot_permuted_sunfish_weights
    plot_permuted_sunfish_weights(epochs=epochs, save_every=save_every, out_path=out_path)

    return R_


def get_sunfish_moves(R_sunfish, boards, depth, out_path, overwrite=False, quiesce=False):
    """
    If the moves have already been calculated for the configuration
    this function just reads the file, otherwise the moves are found
    and pickled.
    :param depth:
    :param out_path:
    :return:
    """
    def step(board, R, depth, quiesce):
        Q, _, moves = alpha_beta_search(board, R=R, depth=depth, maximize=board.turn, quiesce=quiesce)
        return moves.popleft()

    sunfish_moves_path = os.path.join(out_path, 'sunfish_moves.pkl')
    if not overwrite and os.path.exists(sunfish_moves_path):
        print(f'Loaded saved SUNFISH moves from {sunfish_moves_path}')
        with open(sunfish_moves_path, 'rb') as f:
            moves_sunfish = pickle.load(f)
    else:
        moves_sunfish = Parallel(n_jobs=-2)(delayed(step)(board=board, R=R_sunfish, depth=depth, quiesce=quiesce)
                                     for board in tqdm(boards, desc='Getting Sunfish moves'))
        with open(sunfish_moves_path, 'wb') as f:
            pickle.dump(moves_sunfish, f)

    return boards, moves_sunfish


if __name__ == '__main__':
    if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
        print(os.getcwd())
        os.chdir('../')
    from project import policy_walk_multi as policy_walk
    from project import get_midgame_boards, piece, download_lichess_pgn

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
    quiesce = config_data['quiesce']

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
                     quiesce=quiesce,
                     epochs=epochs,
                     delta=delta)
