import copy
import os
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
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board, sunfish_move_to_str, \
    check_moved_same_color
from irl_chess.visualizations import char_to_idxs

from irl_chess.misc_utils.utils import reformat_list
from irl_chess.misc_utils.load_save_utils import process_epoch
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move
from irl_chess.stat_tools.stat_tools import wilson_score_interval


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


def sunfish_move(state, pst, time_limit, move_only=False, run_at_least=1, ):
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


def run_sunfish_GRW(chess_boards, player_moves, config_data, out_path, validation_set):
    if config_data['move_function'] == "sunfish_move":
        use_player_move = False
    elif config_data['move_function'] == "player_move":
        use_player_move = True
    else:
        raise Exception(f"The move function {config_data['move_function']} is not implemented yet")

    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_start'])
    Rs = [R]
    sunfish_boards = [board2sunfish(board, eval_pos(board, R)) for board in chess_boards]
    player_moves_sunfish = [str_to_sunfish_move(move, not board.turn) for move, board in
                            zip(player_moves, chess_boards)]
    delta = config_data['delta']
    start_time = time()

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        actions_true = player_moves_sunfish if use_player_move else parallel(
            delayed(sunfish_move)(state, pst, config_data['time_limit'], True)
            for state in tqdm(sunfish_boards, desc='Getting true Sunfish moves', ))
        for epoch in tqdm(range(config_data['epochs']), desc='Epochs'):
            weight_path = join(out_path, f'weights/{epoch}.csv')
            if os.path.exists(weight_path):
                df = pd.read_csv(weight_path)
                R = df['Result'].values.flatten()
                Rs.append(R)
                print(f'Results loaded for epoch {epoch + 1}, continuing')
                continue

            add = np.zeros(6)
            add[permute_idxs] = np.random.choice([-delta, delta], len(permute_idxs))
            R_new = R + add

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            actions_new = parallel(delayed(sunfish_move)(board, pst_new, config_data['time_limit'], True)
                                   for board in tqdm(sunfish_boards, desc='Getting new Sunfish actions'))
            
            # check sunfish moves same color as player
            check_moved_same_color(sunfish_boards, player_moves_sunfish, actions_new)

            acc = sum([a == a_new for a, a_new in zip(actions_true, actions_new)]) / config_data['n_boards']
            if acc >= last_acc:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))
            Rs.append(R)

            if epoch % config_data['decay_step'] == 0 and epoch != 0:
                delta *= config_data['decay']

            process_epoch(R, epoch, config_data, out_path)

            print(f'Current sunfish accuracy: {acc}, best: {last_acc}')
            print(f'Best R: {R}')
            if time() - start_time > config_data['max_hours'] * 60 * 60:
                print(f'Reached time limit, exited at epoch {epoch}')
                break

            if (epoch + 1) % config_data['val_every'] == 0:
                pst_val = get_new_pst(R)
                val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves=use_player_move, name=epoch)
        pst_val = get_new_pst(R)
        out = val_util(validation_set, out_path, config_data, parallel, pst_val, name=epoch, use_player_moves=use_player_move)
        return out


def val_sunfish_GRW(validation_set, out_path, config_data, epoch, use_player_moves, name=''):
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        df = pd.read_csv(join(out_path, 'weights', f'{epoch}.csv'))
        R = df['Result'].values.flatten()
        pst_val = get_new_pst(R)
        return val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves,name)


def val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves, name=''):
    actions_val = parallel(
        delayed(sunfish_move)(board2sunfish(board, eval_pos(board, None)), pst_val, config_data['time_limit'], True)
        for board, move in tqdm(validation_set, desc='Getting True Sunfish actions'))
    actions_true = parallel(
        delayed(sunfish_move)(board2sunfish(board, eval_pos(board, None)), pst, config_data['time_limit'], True)
        for board, move in tqdm(validation_set, desc='Getting Sunfish validation actions')) if not use_player_moves else actions_val

    acc_temp_true, acc_temp_player = [], []
    actions_val_san, actions_true_san = [], []

    for (state, a_player), a_val, a_true in zip(validation_set, actions_val, actions_true):
        a_val = sunfish_move_to_str(a_val, not state.turn)
        a_true = sunfish_move_to_str(a_true, not state.turn)

        actions_val_san.append(a_val)
        actions_true_san.append(a_true)

        acc_temp_true.append(str(a_true) == a_val)
        acc_temp_player.append(str(a_player) == a_val)

    acc_true = sum(acc_temp_true) / len(acc_temp_true)
    acc_player = sum(acc_temp_player) / len(acc_temp_player)
    if not use_player_moves:
        print(f'Validation accuracy on sunfish: {acc_true}')
    print(f'Validation accuracy on player moves: {acc_player}')

    df = pd.DataFrame([(state, a_player, a_true, a_val) for (state, a_player), a_val, a_true in
                       zip(validation_set, actions_val_san, actions_true_san)],
                      columns=['board', 'a_player', 'a_true' if use_player_moves else 'a_val_copy', 'a_val'])
    os.makedirs(join(out_path, f'validation_output'), exist_ok=True)
    df.to_csv(join(out_path, f'validation_output', f'{name}_{acc_true}.csv'), index=False)
    return acc_player, wilson_score_interval(sum(acc_temp_player), len(acc_temp_player))
