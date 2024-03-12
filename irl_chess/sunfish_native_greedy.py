import json
import os
from os.path import join

import copy
from shutil import copy2

import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from irl_chess import Searcher, Position, initial, sunfish_move, sunfish_move_to_str, pst, pst_only, piece
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish_move_to_str, render, sunfish2board


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


def plot_Rs(Rs, config_data):
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    R_true = np.array(config_data['R_true'])
    target_idxs = config_data['target_idxs']
    Rs = np.array(Rs)
    targets = R_true[target_idxs]
    target_colors = [colors[idx] for idx in target_idxs]
    plt.plot(Rs[:, :-1])
    plt.hlines(targets, 0, Rs.shape[0]-1, colors=target_colors, linestyle='--')
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'))
    plt.show()


# actions = []
# for state in tqdm(states):
#     move = sunfish_move_mod(state, pst_new,time_limit,True)
#     actions.append(move)

if __name__ == '__main__':
    print(os.getcwd())
    if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
        os.chdir('../')
        print(os.getcwd())
    from irl_chess import get_midgame_boards, piece, load_lichess_dfs, create_sunfish_path, plot_R_weights

    path_config = join(os.getcwd(), 'experiment_configs', 'sunfish_native_greedy', 'config.json')
    with open(path_config, 'r') as file:
        config_data = json.load(file)
        path_result = join(os.getcwd(), 'models', 'sunfish_permuted_native')
        out_path = create_sunfish_path(config_data, path_result)
        os.makedirs(out_path, exist_ok=True)
        copy2(path_config, join(out_path, 'config.json'))

    pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
    games = []
    for i in range(1000):
        games.append(chess.pgn.read_game(pgn))

    n_games_mid = 200
    n_games_end = 50
    n_games_total = n_games_mid + n_games_end
    states_boards_mid = [get_board_after_n(game, 15) for game in games[:n_games_mid]]
    states_boards_end = [get_board_after_n(game, 25) for game in games[:n_games_end]]
    states_boards = states_boards_mid + states_boards_end
    states = [board2sunfish(board, eval_pos(board)) for board in states_boards]

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
    decay_step = config_data['decay_step']
    R_noisy_vals = config_data['R_noisy_vals']
    n_boards = config_data['n_boards']
    target_idxs = config_data['target_idxs']
    R_start = np.array(config_data['R_start'])

    last_acc = 0
    accuracies = []
    R_true = np.array([val for val in piece.values()]).astype(float)
    Rs = []
    R = R_start
    with Parallel(n_jobs=n_threads) as parallel:
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states, desc='Getting true moves',))
        for epoch in range(epochs):
            print(f'Epoch {epoch+1}\n', '-' * 25)
            if permute_all:
                add = np.random.uniform(low=-delta, high=delta, size=permute_end_idx-permute_start_idx).astype(R.dtype)
                R_new[permute_start_idx:permute_end_idx] += add
            else:
                add = np.zeros(6)
                add[target_idxs] = np.random.choice([-delta, delta], len(target_idxs))
                R_new = R + add

            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / n_boards
            change_weights = np.random.rand() < np.exp(acc)/(np.exp(acc) + np.exp(last_acc)) if config_data['version'] == 'v1_native_multi' else acc >= last_acc
            if change_weights:
                print(f'Changed weights!')
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)
            accuracies.append((acc, last_acc))
            Rs.append(R)

            if epoch % decay_step == 0 and epoch != 0:
                delta *= decay

            if save_every is not None and save_every and epoch % save_every == 0:
                pd.DataFrame(R.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'),
                                                                             index=False)
            if plot_every is not None and plot_every and epoch % plot_every == 0  and epoch != 0:
                plot_Rs(Rs, config_data)

            print(f'Current accuracy: {acc}, best: {last_acc}')
