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
from irl_chess.chess_utils.alpha_beta_utils import alpha_beta_search_k
from irl_chess import get_midgame_boards, piece, load_lichess_dfs
from irl_chess.visualizations import char_to_idxs, idxs_to_char


def run_sun(df,
            R_sunfish,
            config_data,
            path_result=None,
            ):
    """

    :param df:
    :param min_elo:
    :param max_elo:
    :param n_boards:
    :param depth:
    :return:
    """
    # Full unpack in order to cut and paste:
    n_files = config_data['n_files']
    min_elo = config_data['min_elo']
    max_elo = config_data['max_elo']
    delta = config_data['delta']
    n_boards = config_data['n_boards']
    depth = config_data['search_depth']
    k = config_data['k']
    epochs = config_data['epochs']
    save_every = config_data['save_every']
    permute_all = config_data['permute_all']
    R_noisy_vals = config_data['R_noisy_vals']
    overwrite = config_data['overwrite']
    permute_idxs = char_to_idxs(config_data['permute_char'])
    quiesce = config_data['quiesce']
    n_threads = config_data['n_threads']
    plot_every = config_data['plot_every']
    version = config_data['version']

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = create_sunfish_path(config_data, path_result=path_result)
    R_noisy = copy(R_sunfish)
    R_noisy[permute_idxs] = R_noisy_vals  # Keep the pawn constant
    # R_noisy[1:] += np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape[0] - 1)

    boards, _ = get_midgame_boards(df, n_boards, min_elo=min_elo, max_elo=max_elo, sunfish=False)
    n_boards = len(boards)
    print(f'Found {n_boards} boards, setting n_boards to {n_boards}')

    os.makedirs(out_path, exist_ok=True)
    copy2(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'),
          join(out_path, 'config.json'))

    boards, moves_sunfish = get_sunfish_moves(R_sunfish=R_sunfish, boards=boards, depth=depth, out_path=out_path,
                                              overwrite=overwrite, quiesce=quiesce, n_threads=n_threads)
    R_ = policy_walk(R_noisy, boards, moves_sunfish, config_data=config_data, out_path=out_path)

    from irl_chess import plot_R_weights
    plot_R_weights(config_data=config_data, out_path=out_path)

    return R_


def get_sunfish_moves(R_sunfish, boards, depth, out_path, overwrite=False, quiesce=False, n_threads=-2):
    """
    If the moves have already been calculated for the configuration
    this function just reads the file, otherwise the moves are found
    and pickled.
    :param depth:
    :param out_path:
    :return:
    """

    def step(board, R, depth, quiesce):
        Q, _, moves = alpha_beta_search_k(board, k = 1, R=R, depth=depth, maximize=board.turn, quiesce=quiesce)[0]
        return moves.popleft()

    sunfish_moves_path = os.path.join(out_path, 'sunfish_moves.pkl')
    if not overwrite and os.path.exists(sunfish_moves_path):
        print(f'Loaded saved SUNFISH moves from {sunfish_moves_path}')
        with open(sunfish_moves_path, 'rb') as f:
            moves_sunfish = pickle.load(f)
    else:
        moves_sunfish = Parallel(n_jobs=n_threads)(delayed(step)(board=board, R=R_sunfish, depth=depth, quiesce=quiesce)
                                                   for board in tqdm(boards, desc='Getting Sunfish moves'))

        with open(sunfish_moves_path, 'wb') as f:
            pickle.dump(moves_sunfish, f)

    return boards, moves_sunfish


if __name__ == '__main__':
    if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
        print(os.getcwd())
        os.chdir('../')

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'), 'r') as file:
        config_data = json.load(file)
    n_files = config_data['n_files']
    overwrite = config_data['overwrite']
    version = config_data['version']

    if version == 'v0_multi':
        from irl_chess import policy_walk_v0_multi as policy_walk
    elif version == 'v1_multi':
        from irl_chess import policy_walk_multi as policy_walk

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    df = load_lichess_dfs(websites_filepath=websites_filepath,
                          file_path_data=file_path_data,
                          n_files=n_files,
                          overwrite=overwrite)

    R_sunfish = np.array([val for val in piece.values()]).astype(float)
    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')

    result = run_sun(df,
                     R_sunfish=R_sunfish,
                     path_result=path_result,
                     config_data=config_data
                     )
