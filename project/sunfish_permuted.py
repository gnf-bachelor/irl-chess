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

if 'TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode':
    print("Running in VS Code, fixing sys path")
    import sys

    sys.path.append("./")
from project.chess_utils.utils import alpha_beta_search


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
    epochs = config_data['epochs']
    save_every = config_data['save_every']
    permute_all = config_data['permute_all']
    R_noisy_vals = config_data['R_noisy_vals']
    overwrite = config_data['overwrite']
    permute_start_idx = config_data['permute_start_idx']
    permute_end_idx = config_data['permute_end_idx']
    quiesce = config_data['quiesce']
    n_threads = config_data['n_threads']
    plot_every = config_data['plot_every']
    version = config_data['version']

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = create_sunfish_path(config_data, path_result=path_result)
    R_noisy = copy(R_sunfish)
    R_noisy[permute_start_idx:permute_end_idx] = R_noisy_vals  # Keep the pawn constant
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

    from project import plot_permuted_sunfish_weights
    plot_permuted_sunfish_weights(config_data=config_data, out_path=out_path)

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
        Q, _, moves = alpha_beta_search(board, R=R, depth=depth, maximize=board.turn, quiesce=quiesce)
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


def create_sunfish_path(config_data, path_result):
    min_elo = config_data['min_elo']
    max_elo = config_data['max_elo']
    delta = config_data['delta']
    n_boards = config_data['n_boards']
    search_depth = config_data['search_depth']
    epochs = config_data['epochs']
    permute_all = config_data['permute_all']
    R_noisy_vals = config_data['R_noisy_vals']
    permute_start_idx = config_data['permute_start_idx']
    permute_end_idx = config_data['permute_end_idx']
    quiesce = config_data['quiesce']
    version = config_data['version']
    out_path = join(path_result,
                    f'{permute_all}-{min_elo}-{max_elo}-{search_depth}-{n_boards}-{delta}-{R_noisy_vals}-{permute_start_idx}-{max(permute_end_idx, 0)}-{quiesce}-{version}')
    return out_path


if __name__ == '__main__':
    if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
        print(os.getcwd())
        os.chdir('../')
    from project import get_midgame_boards, piece, load_lichess_dfs

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation', 'config.json'), 'r') as file:
        config_data = json.load(file)
    n_files = config_data['n_files']
    overwrite = config_data['overwrite']
    version = config_data['version']

    if version == 'v0_multi':
        from project import policy_walk_v0_multi as policy_walk
    elif version == 'v1_multi':
        from project import policy_walk_multi as policy_walk
    elif version == 'v1_default':
        from project import policy_walk as policy_walk
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
