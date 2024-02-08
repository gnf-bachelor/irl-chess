import os
import pickle
from copy import copy

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

    R_noisy = copy(R_sunfish)   # Keep the pawn constant:
    R_noisy[1:] = np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape[0] - 1)

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = join(path_result, f'{elo - incr}-{elo + incr}-{search_depth}-{n_boards}')
    os.makedirs(out_path, exist_ok=True)

    boards, moves_sunfish = get_sunfish_moves(max_search_depth, out_path)
    R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=1., epochs=epochs, save_every=save_every,
                     save_path=out_path)

    return R_


def get_sunfish_moves(max_search_depth, out_path):
    """
    If the moves have already been calculated for the configuration
    this function just reads the file, otherwise the moves are found
    and saved as strings in the SAN format.
    :param max_search_depth:
    :param out_path:
    :return:
    """
    boards, _ = get_midgame_boards(df, n_boards, min_elo=elo - incr, max_elo=elo + incr, sunfish=False)

    sunfish_moves_path = os.path.join(out_path, 'sunfish_moves.csv')
    if os.path.exists(sunfish_moves_path):
        moves_sunfish = list(pd.read_csv(sunfish_moves_path, index_col=None, header=None).values.flatten())
    else:
        moves_sunfish = []
        for board in tqdm(boards, desc='Getting Sunfish moves'):
            move, Q = get_best_move(board, R_sunfish, depth=max_search_depth, san=True)
            moves_sunfish.append(move)
        pd.DataFrame(moves_sunfish).to_csv(sunfish_moves_path, index=None, header=None)

    return boards, moves_sunfish


if __name__ == '__main__':
    # if os.getcwd().split('\\')[-1] != 'irl-chess':
    #     os.chdir('../')
    from project import policy_walk, get_midgame_boards, get_best_move, piece, download_lichess_pgn

    n_files = 1
    elo = 1100
    incr = 100
    n_boards = 2000
    search_depth = 3
    epochs = 1
    save_every = 100

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    datapaths = download_lichess_pgn(websites_filepath, file_path_data, n_files=n_files, overwrite=False)
    df = pd.read_csv(datapaths[0], index_col=None)
    for path in tqdm(datapaths[1:], desc='Contatenating DataFrames'):
        df_ = pd.read_csv(path, index_col=None)
        df = pd.concat((df, df_), axis=0)
    df.dropna(inplace=True)

    R_sunfish = np.array([val for val in piece.values()]).astype(float)
    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')

    result = run_sun(df, R_sunfish=R_sunfish, elo=elo, incr=incr,
                     max_search_depth=search_depth, n_boards=n_boards,
                     path_result=path_result, save_every=save_every,
                     epochs=epochs)
