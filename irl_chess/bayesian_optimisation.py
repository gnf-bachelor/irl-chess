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
from irl_chess.chess_utils import get_new_pst
from irl_chess.sunfish_native_pw import sunfish_move


def plot_BO_2d(opt, R_true, target_idxs):
    """

    :param opt:
    :param R_true:
    :param target_idxs:
    :return:
    """
    if len(target_idxs) > 2:
        print('Only the first to target indexes are plotted!')
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
    p1_true, p2_true = R_true[target_idxs[:2]]
    ax2.vlines([p1_true], 0, p2_true, color='red', linestyles='--')
    ax2.hlines([p2_true], 0, p1_true, color='red', linestyles='--')
    ax2.scatter(*opt.X.T, color='red', marker='x')
    # save
    plt.show()
    plt.cla()

    accs = -opt.Y.reshape(-1)
    top_acc = np.maximum.accumulate(accs)
    plt.plot(top_acc)
    plt.title('Top accuracies over time')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    # save
    plt.show()
    plt.cla()


# if __name__ == '__main__':
#     if os.getcwd()[-9:] != 'irl-chess':
#         os.chdir('../')
#
#     # SETUP
#     with open(join(os.getcwd(), 'experiment_configs', 'sunfish_native_greedy_local', 'config.json'), 'r') as file:
#         config_data = json.load(file)
#
#     delta = config_data['delta']
#     n_boards_mid = config_data['n_boards_mid']
#     n_boards_end = config_data['n_boards_end']
#     n_boards_total = n_boards_mid + n_boards_end
#     epochs = config_data['epochs']
#     save_every = config_data['save_every']
#     time_limit = config_data['time_limit']
#     decay = config_data['decay']
#     target_idxs = config_data['target_idxs']
#     R_true = np.array(config_data['R_true'])
#     decay_every = config_data['decay_every']
#     plot_every = config_data['plot_every']
#     n_jobs = config_data['n_jobs']
#
#     pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
#     games = []
#     for i in tqdm(range(n_boards_total*3), 'Getting games'):
#         games.append(chess.pgn.read_game(pgn))
#
#     states_boards_mid = [get_board_after_n(game, 15) for game in games[:n_boards_mid]]
#     states_boards_end = [get_board_after_n(game, 25) for game in games[:n_boards_end]]
#     states_boards = states_boards_mid + states_boards_end
#     states = [board2sunfish(board, eval_pos(board)) for board in states_boards]
#
#     save_path = os.path.join(*config_data['save_path'])
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)
#         os.makedirs(os.path.join(save_path, 'plots'))

def run_bayesian_optimisation(states, config_data, out_path):
    from irl_chess
    # RUN
    R_true = np.array(config_data['R_true'])
    target_idxs = char_to_idxs(config_data['permute_char'])
    domain = []
    possible_values = tuple(np.arange(0, 1000, 10, dtype=int))
    for idx in target_idxs:
        piece_name = 'PNBRQK'[idx]
        domain.append({'name': f'{piece_name} value', 'type': 'discrete', 'domain': possible_values})

    R_start = np.array(config_data['R_start'])
    with Parallel(n_jobs=config_data['n_threads']) as parallel:
        print('Getting true moves\n', '-' * 20)
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, config_data['time_limit'], True)
                                for state in tqdm(states))
        def objective_function(x):
            R_new = copy.copy(R_start)
            R_new[target_idxs] = x[0]
            print(f'R_new: {R_new}')
            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(states))
            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / len(states)
            print(f'Acc: {acc}')
            return -acc

        opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, acquisition_type='EI')
        opt.acquisition.exploration_weight = 0.1
        opt.run_optimization(max_iter=50)

        plot_BO_2d(opt, R_true, target_idxs)


def bayesian_model_result_string(config_data):
    delta = config_data['delta']
    time_limit = config_data['time_limit']
    R_true = '_'.join(model_config_data['R_true'])
    R_start = '_'.join(model_config_data['R_start'])
    move_function = model_config_data['move_function']
    return f"{delta}--{time_limit}--{R_start}-{R_true}--{move_function}"


if __name__ == '__main__':
    from irl_chess import pst, char_to_idxs
    from irl_chess.misc_utils import fix_cwd, load_config, union_dicts, create_result_path
    from irl_chess import get_states
    fix_cwd()
    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)

    match config_data[
        'model']:  # Load the model specified in the "base_config" file. Make sure the "model" field is set
        # correctly and that a model_result_string function is defined to properly store the results.
        case "model1":
            model = print("hi")
        case "model1":
            print("hi")

    out_path = create_result_path(base_config_data, model_config_data, bayesian_model_result_string, path_result=None)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data)

    run_bayesian_optimisation(sunfish_boards, config_data, out_path)