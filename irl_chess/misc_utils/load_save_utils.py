import json
import os
from os.path import join
from shutil import copy2
import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm

from irl_chess.data.make_dataset import download_lichess_pgn
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos
from irl_chess.visualizations import char_to_idxs, plot_R_weights

# ================= Load configs and prepare output =================

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

def copy_configs(out_path, model_name):
    assert_cwd()
    path_config = join(os.getcwd(), 'experiment_configs', 'base_config.json')
    path_model_config = join(os.path.dirname(path_config), model_name, 'config.json')
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
    path = path_result if path_result is not None else join(os.getcwd(), 'results', base_config_data['model'])
    out_path = join(path,
                    f"{base_result_string(base_config_data)}---{model_result_string(model_config_data)}")
                        
    os.makedirs(out_path, exist_ok=True)
    if copy_configs_flag: copy_configs(out_path, model_name=base_config_data['model'])
    return out_path

# ================= Loading chess games =================

def get_board_after_n(game, n):
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if i == n:
            break
    return board

def is_valid_game(game, config_data):
    # Add time control check
    try:
        elo_check_white = config_data['min_elo'] < int(game.headers['WhiteElo']) < config_data['max_elo']
        elo_check_black = config_data['min_elo'] < int(game.headers['BlackElo']) < config_data['max_elo']
        # Add 1 to length check to ensure there is a valid move in the position returned
        length_check = len([el for el in game.mainline_moves()]) + 1 > config_data['n_endgame']
        return elo_check_white and elo_check_black and length_check
    except KeyError:
        return False
    except ValueError:
        return False

def get_states(websites_filepath, file_path_data, config_data):
    pgn_paths = download_lichess_pgn(websites_filepath=websites_filepath,
                                     file_path_data=file_path_data,
                                     overwrite=config_data['overwrite'],
                                     n_files=config_data['n_files'])

    chess_boards = []
    i = 0
    while len(chess_boards) < config_data['n_boards']:
        pgn_path = pgn_paths[i]
        progress = 0
        with open(pgn_path) as pgn:
            size = os.path.getsize(pgn_path)
            with tqdm(total=size, desc=f'Looking through file {i}') as pbar:
                while len(chess_boards) < config_data['n_boards']:
                    game = chess.pgn.read_game(pgn)
                    if is_valid_game(game, config_data=config_data):
                        chess_boards.append(get_board_after_n(game, config_data['n_midgame']))
                        chess_boards.append(get_board_after_n(game, config_data['n_endgame']))
                    pbar.update(pgn.tell() - progress)
                    progress = pgn.tell()
                    if size <= progress:
                        break
            i += 1

    sunfish_boards = [board2sunfish(board, eval_pos(board)) for board in chess_boards]
    return sunfish_boards

# ================= Processing results =================

def process_epoch(R, epoch, config_data, out_path, **kwargs):
    # assert not (not config_data['overwrite'] and os.path.exists(join(out_path, f'{epoch}.csv'))), \
    # "Data already exists but configs are set to not overwrite"
        # Overwrite is for downloaded data files. .........
    
    if config_data['save_every'] and epoch % config_data['save_every'] == 0:
        pd.DataFrame(R.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'),
                                                                    index=False)
    if config_data['plot_every'] and epoch % config_data['plot_every'] == 0:
        plot_R_weights(config_data=config_data,
                       out_path=out_path,
                       epoch=epoch,
                       kwargs=kwargs)

def next_available_epoch(out_path):
    epoch = 0
    while True:
        file_path = join(out_path, f"{epoch}.csv")
        if not os.path.exists(file_path):
            return epoch
        epoch += 1

def init_start_epoch(config_data, out_path):
    if config_data['overwrite']:
        config_data['start_epoch'] = 0 
    else: 
        config_data['start_epoch'] = next_available_epoch(out_path)
        print(f"Continuing from weights of last run starting at epoch {config_data['start_epoch']}")

def model_result_string(model_config_data):
    # This function should be implemented by each model to signify how to format the results
    raise NotImplementedError 