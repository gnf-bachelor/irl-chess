import os
from os.path import join
import chess.pgn
import chess.svg
import numpy as np
from tqdm import tqdm
import json
from joblib import Parallel, delayed
from project import pst
from project.chess_utils.sunfish_utils import board2sunfish, eval_pos, get_new_pst, sunfish_move_mod
from project.chess_utils.utils import get_board_after_n, plot_R

if __name__ == '__main__':
    if os.getcwd()[-9:] != 'irl-chess':
        os.chdir('../')

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_native_greedy', 'config.json'), 'r') as file:
        config_data = json.load(file)

    delta = config_data['delta']
    n_boards_mid = config_data['n_boards_mid']
    n_boards_end = config_data['n_boards_end']
    n_boards_total = n_boards_mid + n_boards_end
    epochs = config_data['epochs']
    save_every = config_data['save_every']
    time_limit = config_data['time_limit']
    decay = config_data['decay']
    target_idxs = config_data['target_idxs']
    R_true = np.array(config_data['R_true'])
    decay_every = config_data['decay_every']
    plot_every = config_data['plot_every']

    pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
    games = []
    for i in tqdm(range(n_boards_total*3), 'Getting games'):
        games.append(chess.pgn.read_game(pgn))

    states_boards_mid = [get_board_after_n(game, 15) for game in games[:n_boards_mid]]
    states_boards_end = [get_board_after_n(game, 25) for game in games[:n_boards_end]]
    states_boards = states_boards_mid + states_boards_end
    states = [board2sunfish(board, eval_pos(board)) for board in states_boards]

    save_path = os.path.join(*config_data['save_path'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(os.path.join(save_path, 'plots'))

    R = np.array(config_data['R_start'])
    best_accs = [0]
    Rs = [R]
    with Parallel(n_jobs=-2) as parallel:
        print('Getting true moves\n', '-' * 20)
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states))

        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}\n', '-' * 20)
            add = np.zeros(6)
            add[target_idxs] = np.random.choice([-delta, delta], len(target_idxs))
            R_new = R + add
            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))])/n_boards_total
            if acc >= best_accs[-1]:
                R = R_new
                best_accs.append(acc)
            Rs.append(R)

            if epoch % plot_every == 0 and epoch != 0:
                plot_R(Rs, R_true, target_idxs, save_path, epoch)

            if epoch % decay_every == 0 and epoch != 0:
                delta *= decay

            if epoch % save_every == 0 and epoch != 0:
                pass

            print(f'Current accuracy: {acc}, best: {best_accs[-1]}')
    plot_R(Rs, R_true, target_idxs, save_path, epoch)
    np.save(os.path.join(save_path, 'weights_over_time.npy'), Rs)
