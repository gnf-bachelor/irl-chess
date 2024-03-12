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
from irl_chess.sunfish_native_pw import sunfish_move, process_epoch
from irl_chess.visualizations import plot_BO_2d

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
        actions_true = parallel(delayed(sunfish_move)(state, pst, config_data['time_limit'], True)
                                for state in tqdm(states, desc='Getting true moves'))
        def objective_function(x):
            R_new = copy.copy(R_start)
            R_new[target_idxs] = x[0]
            print(f'R_new: {R_new}')
            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(states))
            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / len(states)
            print(f'Acc: {acc}')
            return -acc

        opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, acquisition_type='EI')
        opt.acquisition.exploration_weight = 0.1
        opt.run_optimization(max_iter=config_data['epochs'])
        plot_path = join(out_path, 'plot')
        os.makedirs(plot_path, exist_ok=True)
        plot_BO_2d(opt, R_true, target_idxs, epoch=config_data['epochs'], plot_path=plot_path)


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
