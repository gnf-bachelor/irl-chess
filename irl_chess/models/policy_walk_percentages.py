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
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move, sunfish_move_to_str, moves_and_Q_from_result
from irl_chess.models.sunfish_GRW import eval_pos, sunfish_move

# Probability of switching, on Q_numerator and Q_denominator
def joint_probability(Qn, Qd, a=1/1000):
    return np.exp(a*(sum(Qn)-sum(Qd)))


if __name__ == '__main__':
    os.chdir('..')
    os.chdir('..')
    with open('data/move_percentages/moves_1000-1200_fixed', 'r') as f:
        moves_dict = json.load(f)

    with open('experiment_configs/GRW_percentages/config.json', 'r') as f:
        config_data = json.load(f)

    np.set_printoptions(suppress=True)

    n_boards = config_data['n_boards']

    states = [board2sunfish(fen, 0) for fen in list(moves_dict.keys())[:n_boards]]
    player_moves = [move_dict for fen, move_dict in list(moves_dict.items())[:n_boards]]
    actions = [max(move_dict, key=lambda k: move_dict[k][0] - (k == 'sum')) for move_dict in player_moves]

    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_start'])
    Rs = [R]
    delta = config_data['delta']
    time_limit = 0.2

    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        pst_new = get_new_pst(R)
        results = parallel(delayed(sunfish_move)(state, pst_new, time_limit, min_depth=2)
                           for state in tqdm(states, desc='Getting new actions'))

        moves_old, Q_old_dicts = moves_and_Q_from_result(results, states)
        Q_old_old = [Q_old_dicts[state][move] for state, move in list(zip(states, moves_old))]  # Q(s,pi,R)

        for epoch in range(config_data['epochs']):
            print(f'Epoch {epoch + 1}\n', '-' * 25)
            add = np.zeros(6)
            add[permute_idxs] = np.random.choice([-delta, delta], len(permute_idxs))
            R_new = R + add

            pst_new = get_new_pst(R_new)  # Sunfish uses only pst table for calculations
            results = parallel(delayed(sunfish_move)(state, pst_new, time_limit, min_depth=2)
                               for state in tqdm(states, desc='Getting new actions'))

            moves_new, Q_new_dicts = moves_and_Q_from_result(results, states)

            Q_actions = [Q_new_dicts[state][action] for state, action in list(zip(states, actions))]  # Q(s,a,R~)
            Q_policy_old = [Q_new_dicts[state][move] for state, move in list(zip(states, moves_old))]  # Q(s,pi,R~)
            Q_policy_new = [Q_new_dicts[state][move] for state, move in list(zip(states, moves_new))]  # Q(s,pi~,R~)

            acc = sum([player_move == move for player_move, move in list(zip(actions, moves_new))]) / n_boards
            print(f'Accuracy for {R_new}: {acc}')

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
