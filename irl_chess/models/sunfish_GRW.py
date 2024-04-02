import copy
from time import time

import chess
import chess.pgn
import chess.svg
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from irl_chess import Searcher, pst, piece
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
    return best_move, searcher.tp_move, searcher.tp_score


def sunfish_native_result_string(model_config_data):
    delta = model_config_data['delta']
    decay = model_config_data['decay']
    decay_step = model_config_data['decay_step']
    time_limit = model_config_data['time_limit']
    permute_all = model_config_data['permute_all']
    R_true = reformat_list(model_config_data['R_true'], '_')
    R_start = reformat_list(model_config_data['R_start'], '_')
    return f"{delta}-{decay}-{decay_step}-{permute_all}-{time_limit}--{R_start}-{R_true}"


def run_sunfish_GRW(sunfish_boards, player_moves, config_data, out_path, ):
    if config_data['move_function'] == "sunfish_move":
        use_player_move = False
    elif config_data['move_function'] == "player_move":
        use_player_move = True
    else:
        raise Exception(f"The move function {config_data['move_function']} is not implemented yet")

    permute_all = config_data['permute_all']
    permute_idxs = char_to_idxs(config_data['permute_char'])

    R_true = np.array(config_data['R_true'])
    R = np.array(config_data['R_start'])
    R_new = copy.copy(R)
    delta = config_data['delta']

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        actions_true = player_moves if use_player_move else parallel(
            delayed(sunfish_move)(state, pst, config_data['time_limit'], True)
            for state in tqdm(sunfish_boards, desc='Getting true moves', ))
        for epoch in tqdm(range(config_data['epochs']), desc='Epoch'):
            if permute_all:
                add = np.random.uniform(low=-delta, high=delta, size=len(permute_idxs)).astype(R.dtype)
                R_new[permute_idxs] = R[permute_idxs] + add
            else:
                choice = np.random.choice(permute_idxs)
                R_new[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            actions_new = parallel(delayed(sunfish_move)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(sunfish_boards, desc='Getting new actions'))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / config_data['n_boards']
            change_weights = np.random.rand() < np.exp(acc) / (np.exp(acc) + np.exp(last_acc)) if config_data[
                                                                                                      'version'] == 'v1_native_multi' else acc >= last_acc
            if change_weights:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))

            if epoch % config_data['decay_step'] == 0 and epoch != 0:
                delta *= config_data['decay']

            process_epoch(R, epoch, config_data, out_path)  # , accuracies=accuracies)

            print(f'Current accuracy: {acc}, {last_acc}')
