import copy
from time import time
import os
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from irl_chess import Searcher, pst, piece
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board, top_k_moves
from irl_chess.visualizations import char_to_idxs
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


def sunfish_move(state, pst, time_limit, max_depth=1000, move_only=False, run_at_least=1):
    """
    Given a state, p-square table and time limit,
    return the sunfish move.
    :param state:
    :param pst:
    :param time_limit:
    :return:
    """
    searcher = Searcher(pst, max_depth=max_depth)
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


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    with open('data/move_percentages/moves_1000-1200', 'r') as f:
        moves_dict = json.load(f)

    n_boards = 1000
    states = [board2sunfish(fen, 0) for fen in list(moves_dict.keys())[:n_boards]]
    player_moves = list(moves_dict.values())[:n_boards]

    with open('experiment_configs/GRW_percentages/config.json', 'r') as f:
        config_data = json.load(f)

    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_start'])
    Rs = [R]
    delta = config_data['delta']

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        for epoch in range(config_data['epochs']):
            print(f'Epoch {epoch + 1}\n', '-' * 25)
            add = np.zeros(6)
            # don't permute on first epoch.
            if not epoch == 0:
                add[permute_idxs] = np.random.choice([-delta, delta], len(permute_idxs))
            R_new = R + add

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            results = parallel(delayed(sunfish_move)(state, pst_new, config_data['time_limit'])
                               for state in tqdm(states, desc='Getting new actions'))

            moves, best_moves_lists, move_dicts = [], [], []
            for (move, best_moves, move_dict) in results:
                moves.append(move)
                best_moves_lists.append(best_moves)
                move_dicts.append(move_dict)

            actions_new = [top_k_moves(move_dict, 5, uci=True) for move_dict in move_dicts]

            acc = 0
            for i in range(n_boards):
                sunfish_moves = actions_new[i]
                player_move_dict = player_moves[i]
                acc += sum([player_move_dict[move][0] for move in sunfish_moves
                            if move in player_move_dict])
            acc /= n_boards

            if acc >= last_acc:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))
            Rs.append(R)

            print(f'Current accuracy: {acc}, best: {last_acc}')
            print(f'Best R: {R}')
