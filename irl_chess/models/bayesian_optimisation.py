import os
import copy
import GPyOpt

import numpy as np
import pandas as pd

from os.path import join
from tqdm import tqdm
from joblib import Parallel, delayed

from irl_chess.chess_utils import get_new_pst
from irl_chess.sunfish_native_pw import sunfish_move, pst
from irl_chess.visualizations import plot_BO_2d, char_to_idxs
from irl_chess.misc_utils import reformat_list


def run_bayesian_optimisation(sunfish_boards, config_data, out_path):
    # RUN
    R_true = np.array(config_data['R_true'])
    target_idxs = char_to_idxs(config_data['permute_char'])
    domain = []
    possible_values = tuple(np.arange(0, 1000, 10, dtype=int))
    for idx in target_idxs:
        piece_name = 'PNBRQK'[idx]
        domain.append({'name': f'{piece_name} value', 'type': 'discrete', 'domain': possible_values})

    R_start = np.array(config_data['R_start'])
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        # actions_true = parallel(delayed(sunfish_move)(state, pst, config_data['time_limit'], True)
        #                         for state in tqdm(sunfish_boards, desc='Getting true moves'))
        actions_true = [sunfish_move(state, pst, config_data['time_limit'], True) for state in tqdm(sunfish_boards, desc='Getting true moves')]

        def objective_function(x):
            R_new = copy.copy(R_start)
            R_new[target_idxs] = x[0]
            print(f'R_new: {R_new}')
            pst_new = get_new_pst(R_new)
            # actions_new = parallel(delayed(sunfish_move)(state, pst_new, config_data['time_limit'], True)
            #                        for state in tqdm(sunfish_boards, desc='Getting new moves'))
            actions_new = [sunfish_move(state, pst_new, config_data['time_limit'], True) for state in tqdm(sunfish_boards, desc='Getting new moves')]

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / len(sunfish_boards)
            print(f'Acc: {acc}')
            return -acc

        opt = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, acquisition_type='EI')
        opt.acquisition.exploration_weight = 0.1
        plot_path = join(out_path, 'plot')
        os.makedirs(plot_path, exist_ok=True)

        for epoch in tqdm(range(config_data['epochs']), desc='Optimising'):
            opt.run_optimization(max_iter=1, verbosity=config_data['optimisation_verbosity'])
            if epoch and epoch % config_data['plot_every'] == 0:
                plot_BO_2d(opt, R_true, target_idxs, plot_path=plot_path, epoch=epoch)
            if epoch % config_data['save_every'] == 0:
                df = pd.DataFrame(np.concatenate((opt.X, opt.Y), axis=-1))
                df.to_csv(join(out_path, f'Results.csv'), index=False)



def bayesian_model_result_string(model_config_data):
    delta = model_config_data['delta']
    time_limit = model_config_data['time_limit']
    R_true = reformat_list(model_config_data['R_true'], '_')
    R_start = reformat_list(model_config_data['R_start'], '_')
    move_function = model_config_data['move_function']
    return f"{delta}--{time_limit}--{R_start}-{R_true}--{move_function}"