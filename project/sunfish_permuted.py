import os
import pickle
from copy import copy

import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt

def run_sun(df,
            R_sunfish,
            search_depth=3,
            min_elo=1000,
            max_elo=1200,
            n_boards=1000,
            delta=20.,
            permute_all=1,
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

    R_noisy = copy(R_sunfish)   # Keep the pawn constant:
    R_noisy[1:] = 0
    # R_noisy[1:] += np.random.normal(loc=0, scale=sd_noise, size=R_sunfish.shape[0] - 1)

    boards, _ = get_midgame_boards(df, n_boards, min_elo=min_elo, max_elo=max_elo, sunfish=False)
    n_boards = len(boards)
    print(f'Found {n_boards} boards, setting n_boards to {n_boards}')

    if path_result is None:
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = join(path_result, f'{permute_all}-{min_elo}-{max_elo}-{search_depth}-{n_boards}-{delta}')
    os.makedirs(out_path, exist_ok=True)

    boards, moves_sunfish = get_sunfish_moves(boards=boards, depth=depth, out_path=out_path)
    R_ = policy_walk(R_noisy, boards, moves_sunfish, delta=delta, epochs=epochs, save_every=save_every,
                     save_path=out_path)

    plot_weights(epochs=epochs, save_every=save_every, out_path=out_path)

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
        moves_sunfish = list(pd.read_csv(sunfish_moves_path, index_col=None, header=None).values.flatten())
    else:
        moves_sunfish = []
        for board in tqdm(boards, desc='Getting Sunfish moves'):
            move, Q = get_best_move(board, R_sunfish, depth=depth, san=True)
            moves_sunfish.append(move)
        pd.DataFrame(moves_sunfish).to_csv(sunfish_moves_path, index=None, header=None)

    return boards, moves_sunfish


def plot_weights(epochs, save_every, out_path, start_idx=0, ignore_king=True):
    weights = []
    X = np.repeat(np.arange(start_idx, epochs, save_every), 6 - ignore_king).reshape((-1, 6 - ignore_king))
    for i in range(start_idx, epochs, save_every):
        path = os.path.join(out_path, f'{i}.csv')
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=None)
            weights.append(df.values.flatten())
        else:
            print(f'Could not find weights at {i}')
    weights = np.array(weights)

    plt.plot(X, np.array(weights)[:, :6-ignore_king])
    plt.title('Sunfish weights over time')
    plt.xlabel('Number of boards seen')
    plt.ylabel('Weight values')
    plt.legend([key for key in piece.keys()])
    plt.savefig(join(out_path, 'weights_over_time.png'))
    plt.show()


if __name__ == '__main__':
    # if os.getcwd().split('\\')[-1] != 'irl-chess':
    #     os.chdir('../')
    from project import policy_walk, get_midgame_boards, get_best_move, piece, download_lichess_pgn

    n_files = 6
    min_elo = 1000
    max_elo = 1200
    delta = 20.
    n_boards = 20
    search_depth = 3
    epochs = 10
    save_every = 1
    permute_all = 0     # 0/1 for true/false so it can be used in the filename

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

    result = run_sun(df,
                     R_sunfish=R_sunfish,
                     min_elo=min_elo,
                     search_depth=search_depth,
                     max_elo=max_elo,
                     depth=search_depth,
                     n_boards=n_boards,
                     permute_all=permute_all,
                     path_result=path_result,
                     save_every=save_every,
                     epochs=epochs)
