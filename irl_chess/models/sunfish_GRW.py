import copy
import pickle
from os.path import join
from time import time

import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from irl_chess import Searcher, pst, piece, plot_R_weights
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board
from irl_chess.visualizations import char_to_idxs

from irl_chess.misc_utils.utils import reformat_list
from irl_chess.misc_utils.load_save_utils import process_epoch
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move


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


def sunfish_move(state, pst, time_limit, move_only=False, run_at_least=1):
    """
    Given a state, p-square table and time limit,
    return the sunfish move.
    :param state:
    :param pst:
    :param time_limit:
    :return:
    """
    searcher = Searcher(pst)
    start = time()
    best_move = None
    count = 0
    count_gamma = 0
    for depth, gamma, score, move in searcher.search([state]):
        count += 1
        if score >= gamma:
            best_move = move
            count_gamma += 1
        if time() - start > time_limit and count_gamma >= 1 and (count >= run_at_least):
            break
    if best_move is None:
        print(f"best move is: {best_move} and count is {count}")
        print(state.board)
    assert best_move is not None, f"No best move found, this probably means an invalid position was passed to the \
                                   searcher"
    if move_only:
        return best_move
    return best_move, searcher.best_moves, searcher.move_dict


def sunfish_native_result_string(model_config_data):
    delta = model_config_data['delta']
    decay = model_config_data['decay']
    decay_step = model_config_data['decay_step']
    time_limit = model_config_data['time_limit']
    permute_all = model_config_data['permute_all']
    R_true = reformat_list(model_config_data['R_true'], '_')
    R_start = reformat_list(model_config_data['R_start'], '_')
    return f"{delta}-{decay}-{decay_step}-{permute_all}-{time_limit}--{R_start}-{R_true}"


def run_sunfish_GRW(sunfish_boards, player_moves, config_data, out_path, validation_set):
    if config_data['move_function'] == "sunfish_move":
        use_player_move = False
    elif config_data['move_function'] == "player_move":
        use_player_move = True
    else:
        raise Exception(f"The move function {config_data['move_function']} is not implemented yet")

    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_start'])
    Rs = [R]
    delta = config_data['delta']
    start_time = time()

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        actions_true = player_moves if use_player_move else parallel(
            delayed(sunfish_move)(state, pst, config_data['time_limit'], True)
            for state in tqdm(sunfish_boards, desc='Getting true moves', ))
        print(f'First 5 true actions: {actions_true[:5]}')
        for epoch in range(config_data['epochs']):
            print(f'Epoch {epoch + 1}\n', '-' * 25)
            add = np.zeros(6)
            add[permute_idxs] = np.random.choice([-delta, delta], len(permute_idxs))
            R_new = R + add

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            actions_new = parallel(delayed(sunfish_move)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(sunfish_boards, desc='Getting new actions'))
            # check sunfish moves same color as player
            for k, pos in enumerate(sunfish_boards):
                player_move_square = player_moves[k].i
                sunfish_move_square = actions_new[k].i
                move_ok = pos.board[player_move_square].isupper() == pos.board[sunfish_move_square].isupper()
                assert move_ok, 'Wrong color piece moved by sunfish'

            acc = sum([a == a_new for a, a_new in zip(actions_true, actions_new)]) / config_data['n_boards']
            if acc >= last_acc:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))
            Rs.append(R)

            if epoch % config_data['decay_step'] == 0 and epoch != 0:
                delta *= config_data['decay']

            process_epoch(R, epoch, config_data, out_path)  # , accuracies=accuracies)

            print(f'First 5 model actions: {actions_new[:5]}')
            print(f'Current accuracy: {acc}, best: {last_acc}')
            print(f'Best R: {R}')
            if time() - start_time > config_data['max_hours'] * 60 * 60:
                break

        pst_val = get_new_pst(R)
        actions_val = parallel(delayed(sunfish_move)(state, pst_val, config_data['time_limit'], True)
                               for state, move in tqdm(validation_set, desc='Getting validation actions'))
        acc_temp = []
        for (state, a), a_val in zip(validation_set, actions_val):
            acc_temp.append(a == a_val)
        acc = sum(acc_temp) / len(acc_temp)
        print(f'Validation accuracy: {acc}')
        df = pd.DataFrame([(state, a_true, a_val) for (state, a_true), a_val in zip(validation_set, actions_val)])
        df.to_csv(join(out_path, 'validation_output.csv'))
