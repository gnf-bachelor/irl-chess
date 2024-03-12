import json
import os
from os.path import join

import copy
from shutil import copy2

import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from irl_chess import Searcher, Position, initial, sunfish_move, sunfish_move_to_str, pst, pst_only, piece
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish_move_to_str, render, sunfish2board
from irl_chess.visualizations import char_to_idxs, plot_permuted_sunfish_weights
from irl_chess import piece, plot_permuted_sunfish_weights, download_lichess_pgn, sunfish_weights
from irl_chess import union_dicts


def assert_cwd():
    assert os.path.basename(os.getcwd()) == 'irl-chess', f"This file {__file__} is not being run from the appopriate\
            directory {'irl-chess'} but instead {os.getcwd()}"

def fix_cwd():
    try:
        assert_cwd()
    except AssertionError:
        print(os.getcwd())
        print("Attempting to fix the current working directory.")
        os.chdir('../')
        print(os.getcwd())
        assert_cwd()

def load_config():
    assert_cwd()
    path_config = join(os.getcwd(), 'experiment_configs', 'base_config.json')
    with open(path_config, 'r') as file:
        base_config_data = json.load(file)
    path_model_config = join(os.path.dirname(path_config), base_config_data["model"], 'config.json')
    with open(path_model_config, 'r') as file:
        model_config_data = json.load(file)
    return base_config_data, model_config_data

def copy_configs(out_path):
    assert_cwd()
    path_config = join(os.getcwd(), 'experiment_configs', 'base_config.json')
    path_model_config = join(os.path.dirname(path_config), base_config_data["model"], 'config.json')
    out_path_config = join(out_path, 'configs')
    os.makedirs(out_path_config, exist_ok=True)
    copy2(path_config, join(out_path_config, 'base_config.json'))
    copy2(path_model_config, join(out_path_config, 'model_config.json'))
    return

def base_result_string(base_config_data):
    time_control =  base_config_data['time_control']
    min_elo =       base_config_data['min_elo']
    max_elo =       base_config_data['max_elo']
    n_midgame =     base_config_data['n_midgame']
    n_endgame =     base_config_data['n_endgame']
    n_boards =      base_config_data['n_boards']
    permute_char =  ''.join(base_config_data['permute_char'])
    return f"{time_control}-{min_elo}-{max_elo}-{n_midgame}_to_{n_endgame}-{n_boards}-{permute_char}"    

def create_result_path(base_config_data, model_config_data, model_result_string, path_result=None, copy_configs_flag=True):
    model = base_config_data['model']

    path = path_result if path_result is not None else join(os.getcwd(), 'models', base_config_data["model"])
    out_path = join(path,
                    f"{base_result_string(base_config_data)}---\
                        {model_result_string(model_config_data)}")
    os.makedirs(out_path, exist_ok=True)
    if copy_configs_flag: copy_configs(out_path)
    return out_path

def run_sunfish_native(sunfish_boards, config_data, out_path):
    permute_all = config_data['permute_all']
    permute_idxs = char_to_idxs(config_data['permute_char'])

    R = np.array(config_data['R_true'])
    R_new = np.array([config_data['R_start']])
    delta = config_data['delta']

    last_acc = 0
    accuracies = []

    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, config_data['time_limit'], True)
                                for state in tqdm(sunfish_boards, desc='Getting true moves', ))
        for epoch in tqdm(range(config_data['epochs']), desc='Epoch'):
            if permute_all:
                add = np.random.uniform(low=-delta, high=delta, size=len(permute_idxs)).astype(R.dtype)
                R_new[permute_idxs] += add
            else:
                choice = np.random.choice(permute_idxs)
                R_new[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

            pst_new = get_new_pst(R_new)    # Sunfish uses only pst table for calculations
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(sunfish_boards, desc='Getting new actions'))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / config_data['n_boards']
            change_weights = np.random.rand() < np.exp(acc) / (np.exp(acc) + np.exp(last_acc)) if config_data['version'] == 'v1_native_multi' else acc >= last_acc
            if change_weights:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))

            if epoch % config_data['decay_step'] == 0 and epoch != 0:
                delta *= config_data['decay']

            process_epoch(R, epoch, config_data, out_path) #, accuracies=accuracies)

            print(f'Current accuracy: {acc}, {last_acc}')


def process_epoch(R, epoch, config_data, out_path, **kwargs):
    if config_data['save_every'] and epoch % config_data['save_every'] == 0:
        pd.DataFrame(R.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'),
                                                                    index=False)
    if config_data['plot_every'] and epoch % config_data['plot_every'] == 0:
        plot_permuted_sunfish_weights(config_data=config_data,
                                      out_path=out_path,
                                      epoch=epoch,
                                      kwargs=kwargs)

def model_result_string(model_config_data):
    return None


if __name__ == '__main__':
    fix_cwd()

    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)
    match config_data['model']:
        case "model1":
            print("hi")
        case "model1":
            print("hi")
    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data)
    run_sunfish_native(sunfish_boards=sunfish_boards, config_data=config_data, out_path=out_path)