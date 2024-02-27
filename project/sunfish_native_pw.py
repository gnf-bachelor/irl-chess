import json
import os
from os.path import join

import copy
import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from project import Searcher, Position, initial, sunfish_move, sunfish_move_to_str, pst, pst_only, piece
from project.chess_utils.sunfish_utils import board2sunfish, sunfish_move_to_str, render, sunfish2board


def get_board_after_n(game, n):
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if i == n:
            break
    return board


def get_board_last(tp_move, init_pos):
    pos_ = init_pos
    while pos_ in tp_move.keys():
        move = tp_move[pos_]
        pos_ = pos_.move(move)
    return pos_


# Assuming white, R is array of piece values
def eval_pos(board, R=None):
    pos = board2sunfish(board, 0)
    pieces = 'PNBRQK'
    if R is not None:
        piece_dict = {p: R[i] for i, p in enumerate(pieces)}
    else:
        piece_dict = piece
    eval = 0
    for row in range(20, 100, 10):
        for square in range(1 + row, 9 + row):
            p = pos.board[square]
            if p == '.':
                continue
            if p.islower():
                p = p.upper()
                eval -= piece_dict[p] + pst[p][119 - square]
            else:
                eval += piece_dict[p] + pst[p][square]
    return eval


def get_new_pst(R):
    assert len(R) == 6
    pieces = 'PNBRQK'
    piece_new = {p: val for p, val in list(zip(pieces, R))}
    pst_new = copy.deepcopy(pst_only)
    for k, table in pst_only.items():
        padrow = lambda row: (0,) + tuple(x + piece_new[k] for x in row) + (0,)
        pst_new[k] = sum((padrow(table[i * 8: i * 8 + 8]) for i in range(8)), ())
        pst_new[k] = (0,) * 20 + pst_new[k] + (0,) * 20
    return pst_new


def sunfish_move_mod(state, pst, time_limit, only_move=False):
    searcher = Searcher(pst)
    if only_move:
        return sunfish_move(searcher, [state], time_limit=time_limit)[0]
    return sunfish_move(searcher, [state], time_limit=time_limit)


def plot_R(Rs, path=None):
    Rs = np.array(Rs)
    plt.plot(Rs[:, :-1], )
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'))
    if path is not None:
        plt.imsave(join(path, 'weights_over_time.png'))
    plt.show()


# actions = []
# for state in tqdm(states):
#     move = sunfish_move_mod(state, pst_new,time_limit,True)
#     actions.append(move)

if __name__ == '__main__':
    # pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
    # games = []
    # for i in range(1000):
    #     games.append(chess.pgn.read_game(pgn))
    # if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
    #     os.chdir('../')
    from project import get_midgame_boards, piece, load_lichess_dfs, create_sunfish_path, plot_permuted_sunfish_weights

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation_native', 'config.json'), 'r') as file:
        config_data = json.load(file)

    n_files = config_data['n_files']
    overwrite = config_data['overwrite']
    version = config_data['version']
    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')
    out_path = join(os.getcwd(), 'models', 'sunfish_permuted_native')
    os.makedirs(out_path, exist_ok=True)

    df = load_lichess_dfs(websites_filepath=websites_filepath,
                          file_path_data=file_path_data,
                          n_files=n_files,
                          overwrite=overwrite)

    boards, _ = get_midgame_boards(df,
                                   n_boards=config_data['n_boards'],
                                   min_elo=config_data['min_elo'],
                                   max_elo=config_data['max_elo'],
                                   sunfish=False,
                                   n_steps=15)

    n_games = 200
    # states_boards = [get_board_after_n(game, 15) for game in games[:n_games]]
    states = [board2sunfish(board, eval_pos(board)) for board in boards]

    epochs = config_data['epochs']
    time_limit = config_data['time_limit']
    delta = config_data['delta']
    save_every = config_data['save_every']
    permute_all = config_data['permute_all']
    permute_start_idx = config_data['permute_start_idx']
    permute_end_idx = config_data['permute_end_idx']
    quiesce = config_data['quiesce']
    n_threads = config_data['n_threads']
    plot_every = config_data['plot_every']
    decay = config_data['decay']
    R_noisy_vals = config_data['R_noisy_vals']

    last_acc = 0
    Rs = []
    R = np.array([val for val in piece.values()]).astype(float)
    R_new = copy.copy(R)
    R_new[permute_start_idx:permute_end_idx] = R_noisy_vals

    with Parallel(n_jobs=n_threads) as parallel:
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states, desc='Getting true moves',))
        for epoch in tqdm(range(epochs), desc='Epoch'):
            if permute_all:
                add = np.random.uniform(low=-delta, high=delta, size=R.shape[0] - 1).astype(R.dtype)
                R_new[permute_start_idx:permute_end_idx] += add
            else:
                choice = np.random.choice(np.arange(permute_start_idx, permute_end_idx))
                R_new[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

            pst_new = get_new_pst(R_new)
            states = [board2sunfish(board, eval_pos(board, R_new)) for board in boards]
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states, desc='Getting new actions'))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / n_games
            if acc >= last_acc:
                R = copy.copy(R_new)
                last_acc = acc
            Rs.append(R)

            if epoch % 10 == 0 and epoch != 0:
                plot_R(Rs)

            if epoch % 30 == 0 and epoch != 0:
                delta *= decay

            if save_every is not None and epoch % save_every == 0:
                pd.DataFrame(R_new.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'),
                                                                             index=False)
            if plot_every is not None and epoch % plot_every == 0:
                plot_permuted_sunfish_weights(config_data=config_data, out_path=out_path, )

            print(f'Current accuracy: {acc}')
    plot_R(Rs)
