import os
from os.path import join
import chess.pgn
import copy
import chess.svg
import GPyOpt
import numpy as np
from tqdm import tqdm
import json
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib as mpl
from project import pst
from project.chess_utils.sunfish_utils import board2sunfish, eval_pos, get_new_pst, sunfish_move_mod
from project.chess_utils.utils import get_board_after_n

def plot_BO_2d(opt):
    piece_one = np.linspace(0,1000,1000)
    piece_two = np.linspace(0, 1000, 1000)
    pgrid = np.array(np.meshgrid(piece_one, piece_two, indexing='ij'))
    # we then unfold the 4D array and simply pass it to the acqusition function
    acq_img = opt.acquisition.acquisition_function(pgrid.reshape(2, -1).T)
    acq_img = (-acq_img - np.min(-acq_img)) / (np.max(-acq_img - np.min(-acq_img)))
    acq_img = acq_img.reshape(pgrid[0].shape[:2])
    mod_img = -opt.model.predict(pgrid.reshape(2, -1).T)[0]
    mod_img = mod_img.reshape(pgrid[0].shape[:2])

    fig, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(acq_img.T, origin='lower')
    ax1.set_xlabel('piece_one')
    ax1.set_ylabel('piece_two')
    ax1.set_title('Acquisition function')
    ax2.imshow(mod_img.T, origin='lower')
    ax2.set_xlabel('piece_one')
    ax2.set_ylabel('piece_two')
    ax2.set_title('Model')
    ax2.scatter(*opt.X.T, c=np.arange(len(opt.X)), cmap='Reds', marker='x')
    plt.show()

    accs = -opt.Y.reshape(-1)
    top_acc = np.maximum.accumulate(accs)

if __name__ == '__main__':
    if os.getcwd()[-9:] != 'irl-chess':
        os.chdir('../')

    # SETUP
    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_native_greedy_local', 'config.json'), 'r') as file:
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
    n_jobs = config_data['n_jobs']

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

    # RUN
    domain = []
    for idx in target_idxs:
        piece_name = 'PNBRQK'[idx]
        domain.append({'name': f'{piece_name} value', 'type': 'continuous', 'domain': (0, 1000)})

    R_start = np.array(config_data['R_start'])
    with Parallel(n_jobs=n_jobs) as parallel:
        print('Getting true moves\n', '-' * 20)
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states))
        def objective_function(x):
            R_new = copy.copy(R_start)
            R_new[target_idxs] = x[0]
            print(f'R_new: {R_new}')
            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states))
            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / n_boards_total
            print(f'Acc: {acc}')
            return -acc


        opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, acquisition_type='EI')
        opt.acquisition.exploration_weight = 0.5
        opt.run_optimization(max_iter=20)
        plot_BO_2d(opt)
