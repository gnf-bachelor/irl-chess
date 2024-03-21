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
from project import pst
from project.chess_utils.sunfish_utils import board2sunfish, eval_pos, get_new_pst, sunfish_move_mod
from project.chess_utils.utils import get_board_after_n

def plot_BO_2d(opt, R_true, target_idxs):
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
    p1_true, p2_true = R_true[target_idxs]
    ax2.vlines([p1_true], 0, p2_true, color='red', linestyles='--')
    ax2.hlines([p2_true], 0, p1_true, color='red', linestyles='--')
    ax2.scatter(*opt.X.T, color='red', marker='x')
    plt.show()

    accs = -opt.Y.reshape(-1)
    top_acc = np.maximum.accumulate(accs)
    plt.plot(top_acc)
    plt.title('Top accuracies over time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.show()

def plot_R_BO(opt, R_true, target_idxs, epoch=None, save_path=False):
    target_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    x = np.array([R_true[:-1]] * len(opt.Y))
    x[:, target_idxs] = opt.X
    c = np.hstack((x, -opt.Y))
    cumulative_argmax = np.array([c[np.argmax(c[:i + 1, -1])] for i in range(len(c))])
    for i, values in enumerate(cumulative_argmax[:, :-1].T):
        plt.plot(values, c=target_colors[i])
    plt.hlines(R_true[:-1],0, c.shape[0]-1, colors=target_colors, linestyle='--')
    plt.suptitle('Bayesian Optimisation')
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'), loc='lower right')
    if save_path:
        plt.savefig(os.path.join(save_path, f'weights_over_time_{epoch}.png'))
    plt.show()


if __name__ == '__main__':
    if os.getcwd()[-9:] != 'irl-chess':
        os.chdir('../')

    # SETUP
    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_native_bo', 'config.json'), 'r') as file:
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
    n_games = 0
    print('Getting games')
    while len(games) < n_boards_total:
        game = chess.pgn.read_game(pgn)
        if len(list(game.mainline_moves())) > 26:
            games.append(game)

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
    possible_values = tuple(np.arange(0, 1000, 10, dtype=int))
    for idx in target_idxs:
        piece_name = 'PNBRQK'[idx]
        domain.append({'name': f'{piece_name} value', 'type': 'discrete', 'domain': possible_values})

    R_start = np.array(config_data['R_start'])
    with Parallel(n_jobs=n_jobs) as parallel:
        print('Getting true moves\n', '-' * 20)
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states))
        def objective_function(x):
            R_new = copy.copy(R_start)
            R_new[target_idxs] = x
            print(f'R_new: {R_new}')
            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states))
            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / n_boards_total
            print(f'Acc: {acc}')
            return -acc

        opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, acquisition_type='EI')
        opt.acquisition.exploration_weight = 0.2
        opt.run_optimization(max_iter=epochs)
        plot_R_BO(opt, R_true, target_idxs)
        plot_BO_2d(opt, R_true, target_idxs)
