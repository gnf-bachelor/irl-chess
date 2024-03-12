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
from irl_chess.visualizations import char_to_idxs


def get_board_after_n(game, n):
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if i == n:
            break
    return board


def get_board_last(tp_move, init_pos):
    pos_ = init_pos
    while pos_ in tp_move.keys():
        move = tp_move[pos_]
        pos_ = pos_.move(move)
    return pos_


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


def get_new_pst(R):
    assert len(R) == 6
    pieces = 'PNBRQK'
    piece_new = {p: val for p, val in list(zip(pieces, R))}
    pst_new = copy.deepcopy(pst_only)
    for k, table in pst_only.items():
        padrow = lambda row: (0,) + tuple(x + piece_new[k] for x in row) + (0,)
        pst_new[k] = sum((padrow(table[i * 8: i * 8 + 8]) for i in range(8)), ())
        pst_new[k] = (0,) * 20 + pst_new[k] + (0,) * 20
    return pst_new


def sunfish_move_mod(state, pst, time_limit, only_move=False):
    searcher = Searcher(pst)
    if only_move:
        return sunfish_move(searcher, [state], time_limit=time_limit)[0]
    return sunfish_move(searcher, [state], time_limit=time_limit)


def plot_R(Rs, path=None):
    Rs = np.array(Rs)
    plt.plot(Rs[:, :-1], )
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'))
    if path is not None:
        plt.imsave(join(path, 'weights_over_time.png'))
    plt.show()


# actions = []
# for state in tqdm(states):
#     move = sunfish_move_mod(state, pst_new,time_limit,True)
#     actions.append(move)


def is_valid_game(game, config_data):
    # Add time control check
    try:
        elo_check_white = config_data['min_elo'] < int(game.headers['WhiteElo']) < config_data['max_elo']
        elo_check_black = config_data['min_elo'] < int(game.headers['BlackElo']) < config_data['max_elo']
        length_check = len([el for el in game.mainline_moves()]) < config_data['n_endgame']
    except KeyError:
        return False
    except ValueError:
        return False
    return elo_check_white and elo_check_black and length_check


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


# ========================== The start of many run_model specific functions =============================

def union_dicts(dict1, dict2):
    # Check for common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Error: Dictionaries have common keys: {common_keys}")

    # If no common keys, perform the union
    return {**dict1, **dict2}


def assert_cwd():
    assert os.path.basename(os.getcwd()) == 'irl-chess', f"This file {__file__} is not being run from the appopriate\
        directory {'irl-chess'} but instead {os.getcwd()}"


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


def model_result_string(model_config_data):
    return None


if __name__ == '__main__':
    print(os.getcwd())
    if os.path.basename(os.getcwd()) != 'irl-chess':
        os.chdir('../')
        print(os.getcwd())
    from irl_chess import piece, plot_permuted_sunfish_weights, download_lichess_pgn

    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)

    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data)

    permute_all = config_data['permute_all']
    permute_idxs = char_to_idxs(config_data['permute_char'])

    last_acc = 0
    accuracies = []
    R = np.array([val for val in piece.values()]).astype(float)
    R_new = copy.copy(R)
    R_new[permute_idxs] = config_data['R_noisy_vals']
    delta = config_data['delta']

    with Parallel(n_jobs=config_data['n_threads']) as parallel:
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, config_data['time_limit'], True)
                                for state in tqdm(sunfish_boards, desc='Getting true moves', ))
        for epoch in tqdm(range(config_data['epochs']), desc='Epoch'):
            if permute_all:
                add = np.random.uniform(low=-delta, high=delta, size=len(permute_idxs)).astype(R.dtype)
                R_new[permute_idxs] += add
            else:
                choice = np.random.choice(permute_idxs)
                R_new[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

            pst_new = get_new_pst(R_new)
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, config_data['time_limit'], True)
                                   for state in tqdm(sunfish_boards, desc='Getting new actions'))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))]) / config_data['n_boards']
            change_weights = np.random.rand() < np.exp(acc) / (np.exp(acc) + np.exp(last_acc)) if config_data[
                                                                                                      'version'] == 'v1_native_multi' else acc >= last_acc
            if change_weights:
                R = copy.copy(R_new)
                last_acc = copy.copy(acc)
            accuracies.append((acc, last_acc))

            if epoch % config_data['decay_step'] == 0 and epoch != 0:
                delta *= config_data['decay']

            if config_data['save_every'] is not None and config_data['save_every'] and epoch % config_data[
                'save_every'] == 0:
                pd.DataFrame(R.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'),
                                                                            index=False)
            if config_data['plot_every'] is not None and config_data['plot_every'] and epoch % config_data[
                'plot_every'] == 0:
                plot_permuted_sunfish_weights(config_data=config_data,
                                              out_path=out_path,
                                              epoch=epoch,
                                              accuracies=accuracies)

            print(f'Current accuracy: {acc}, {last_acc}')
