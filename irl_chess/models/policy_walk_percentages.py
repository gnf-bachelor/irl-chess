import copy
from time import time
import os
import random
import json
import chess
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from irl_chess import Searcher, pst, piece
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board, top_k_moves
from irl_chess.visualizations import char_to_idxs
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move, sunfish_move_to_str

'''
Nuværende problem: der er stadig visse player moves som sunfish ikke fanger, dvs
sunfish gemmer dem ikke i sin search. Mulige løsninger:
- drop dem, og læg gennemsnittet til
'''


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

def moves_and_Q_from_result(results, states):
    moves, _, Q_new_dicts = [], [], {}
    for state, (move, best_moves, move_dict) in list(zip(states, results)):
        if move is not None:
            moves.append(sunfish_move_to_str(move))
            Q_new_dicts[state] = {sunfish_move_to_str(move): scores[-1] for move, scores in move_dict.items()
                                  if move is not None}
    return moves, Q_new_dicts

# Probability of switching, on Q_numerator and Q_denominator
def joint_probability(Qn, Qd, a=1/1000):
    return np.exp(a*(sum(Qn)-sum(Qd)))


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    with open('data/move_percentages/moves_1000-1200_fixed', 'r') as f:
        moves_dict = json.load(f)

    with open('experiment_configs/GRW_percentages/problem_idxs.json', 'r') as f:
        problem_idxs = json.load(f)
    problem_fens = []
    for i, (pos, moves) in enumerate(moves_dict.items()):
        if i in problem_idxs:
            problem_fens.append(pos)
    for fen in problem_fens:
        del moves_dict[fen]

    n_boards = 1000

    states = [board2sunfish(fen, 0) for fen in list(moves_dict.keys())[:n_boards]]
    player_moves = [move_dict for fen, move_dict in list(moves_dict.items())[:n_boards]]
    actions = [max(move_dict, key=lambda k: move_dict[k][0] - (k == 'sum')) for move_dict in player_moves]


    with open('experiment_configs/GRW_percentages/config.json', 'r') as f:
        config_data = json.load(f)

    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_start'])
    Rs = [R]
    delta = config_data['delta']
    time_limit = 1

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        pst_new = get_new_pst(R)
        results = parallel(delayed(sunfish_move)(state, pst_new, time_limit)
                           for state in tqdm(states, desc='Getting new actions'))

        moves_old, Q_old_dicts = moves_and_Q_from_result(results, states)
        Q_old_old = [Q_old_dicts[state][move] for state, move in list(zip(states, moves_old))]  # Q(s,pi,R)
        problem_set = set()
        # Check legal moves
        for i, (state, action) in enumerate(list(zip(states, actions))):
            Q_dict = Q_old_dicts[state]
            sunfish_moves = set(Q_dict.keys())
            if action not in sunfish_moves:
                print(f'{action} not in {sunfish_moves}')
                print(i)
                problem_set.add(i)

        for epoch in range(config_data['epochs']):
            print(f'Epoch {epoch + 1}\n', '-' * 25)
            add = np.zeros(6)
            add[permute_idxs] = np.random.choice([-delta, delta], len(permute_idxs))
            R_new = R + add

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            results = parallel(delayed(sunfish_move)(state, pst_new, time_limit)
                               for state in tqdm(states, desc='Getting new actions'))

            moves_new, Q_new_dicts = moves_and_Q_from_result(results, states)

            # Check legal moves
            for i, (state, action) in enumerate(list(zip(states, actions))):
                Q_dict = Q_new_dicts[state]
                sunfish_moves = set(Q_dict.keys())
                if action not in sunfish_moves:
                    print(f'{action} not in {sunfish_moves}')
                    print(i)
                    problem_set.add(i)
            continue
            ####

            Q_actions = [Q_new_dicts[state][action] for state, action in list(zip(states, actions))]  # Q(s,a,R~)
            Q_policy_old = [Q_new_dicts[state][move] for state, move in list(zip(states, moves_old))]  # Q(s,pi,R~)
            Q_policy_new = [Q_new_dicts[state][move] for state, move in list(zip(states, moves_new))]  # Q(s,pi~,R~)

            if any([q_a > q_p for q_a, q_p in list(zip(Q_actions, Q_policy_new))]):
                joint_p_fraction = joint_probability(Q_policy_new, Q_old_old)
                p = min(1, joint_p_fraction)
                if p > random.random():
                    print(f'Changed weights! From {R}\n to {R_new}\n Probability was: {joint_p_fraction}')
                    moves_old = copy.copy(moves_new)
                    Q_old_old = copy.copy(Q_policy_new)
                    R = R_new
                    continue

            else:
                joint_p_fraction = joint_probability(Q_policy_old, Q_old_old)
                p = min(1, joint_p_fraction)
                if p > random.random():
                    print(f'Changed weights! Probability was: {joint_p_fraction}')
                    R = R_new
                    continue

            print(f'Weights unchanged, p was {joint_p_fraction}')
