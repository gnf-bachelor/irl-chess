import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from os.path import join


def run_sun(df,
            R_sunfish,
            elo=1100,
            incr=100,
            n_boards=1000,
            sd_noise=50,
            epochs=1,
            max_search_depth=3,
            path_result=None,
            save_every=1000):
    """

    :param df:
    :param elo:
    :param incr:
    :param n_boards:
    :param sd_noise:
    :param max_search_depth:
    :return:
    """
    moves_sunfish = []
    boards, moves = get_midgame_boards(df, n_boards, min_elo=elo - incr, max_elo=elo + incr, sunfish=False)

    for board in tqdm(boards, desc='Getting Sunfish moves'):
        move, Q = get_best_move(board, R_sunfish, depth=max_search_depth)
        moves_sunfish.append(move)

    R_noisy = R_sunfish + np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape)
    if path_result is not None:
        out_path = join(path_result, f'{elo - incr}-{elo + incr}-{search_depth}-{n_boards}')
        os.makedirs(out_path, exist_ok=True)
        R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=1., epochs=epochs, save_every=save_every,
                         save_path=out_path)
    else:
        R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=1., epochs=epochs)

    return R_


if __name__ == '__main__':
    if os.getcwd().split('\\')[-1] != 'irl-chess':
        os.chdir('../')
    from project import policy_walk, get_midgame_boards, get_best_move, piece, download_lichess_pgn

    n_files = 2
    elo = 1100
    incr = 100
    n_boards = 200
    search_depth = 3
    save_every = 1000

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    datapaths = download_lichess_pgn(websites_filepath, file_path_data, n_files=n_files, overwrite=False)
    df = pd.read_csv(datapaths[0], index_col=None)
    R_sunfish = np.array([val for val in piece.values()])
    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')

    result = run_sun(df, R_sunfish=R_sunfish, elo=elo, incr=incr,
                     max_search_depth=search_depth, n_boards=n_boards,
                     path_result=path_result, save_every=save_every)
