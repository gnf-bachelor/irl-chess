import json
import os
import pickle
import re
from collections import defaultdict
from os.path import join
from shutil import copy2
from copy import deepcopy
import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from irl_chess.chess_utils.sunfish_utils import board2sunfish, str_to_sunfish_move
from irl_chess.data.make_dataset import download_lichess_pgn
from irl_chess.visualizations import char_to_idxs, plot_R_weights, load_weights_epoch
from irl_chess.misc_utils.utils import reformat_list


# ================= Load configs and prepare output =================
pattern_time = r'%clk (\d+):(\d+):(\d+)'


def assert_cwd():
    assert os.path.basename(os.getcwd()) == 'irl-chess', f"This file {__file__} is not being run from the appropriate\
            directory {'irl-chess'} but instead {os.getcwd()}"


def fix_cwd():
    try:
        assert_cwd()
    except AssertionError:
        print(os.getcwd())
        print("Attempting to fix the maia_pretrained working directory.")
        os.chdir('../')
        print(os.getcwd())
        assert_cwd()


def load_config(model=None):
    assert_cwd()
    path_config = join(os.getcwd(), 'experiment_configs', 'base_config.json')
    with open(path_config, 'r') as file:
        base_config_data = json.load(file)
    if model is not None: base_config_data["model"] = model
    path_model_config = join(os.path.dirname(path_config), base_config_data["model"], 'config.json')
    with open(path_model_config, 'r') as file:
        model_config_data = json.load(file)
    define_permute_idxs(base_config_data)
    return base_config_data, model_config_data

def load_model_functions(config_data):
    match config_data[
        'model']:  # Load the model specified in the "base_config" file. Make sure the "model" field is set
        # correctly and that a model_result_string function is defined to properly store the results.
        case "sunfish_GRW":  # Sunfish Greedy Random Walk
            from irl_chess.models.sunfish_GRW import run_sunfish_GRW as model, \
                sunfish_native_result_string as model_result_string
        case "bayesian_optimisation":
            from irl_chess.models.bayesian_optimisation import run_bayesian_optimisation as model, \
                bayesian_model_result_string as model_result_string
        case "maia_pretrained":
            from irl_chess.models.maia_pretrained import run_maia_pre as model, \
                maia_pre_result_string as model_result_string
        case "BIRL":
            from irl_chess.models.BIRL import run_BIRL as model, \
                BIRL_result_string as model_result_string
        case _:
            raise Exception(f"No model found with the name {config_data['model']}")
    return model, model_result_string


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
    time_control = base_config_data['time_control']
    min_elo = base_config_data['min_elo']
    max_elo = base_config_data['max_elo']
    n_midgame = base_config_data['n_midgame']
    n_endgame = base_config_data['n_endgame']
    n_boards = base_config_data['n_boards']
    move_function = base_config_data['move_function']
    permute_char = ''.join(base_config_data['permute_char'])
    permute_pst_char = ''.join(base_config_data['permute_pst_char']) # Piece square table weights as parameters
    PA_KS_PS = ''.join([heuristic for heuristic, use_heuristic in zip(["PA", "KS", "PS"], base_config_data['include_PA_KS_PS']) if use_heuristic]) # Piece activity, King safety, Pawn structure
    permute_how_many = base_config_data['permute_how_many'] if base_config_data['permute_how_many'] != -1 else "all"
    max_hours = str(base_config_data['max_hours'])
    RP_start = reformat_list(base_config_data['RP_start'], '_')
    RP_true = reformat_list(base_config_data['RP_true'], '_')
    Rpst_start = reformat_list(base_config_data['Rpst_start'], '_')
    Rpst_true = reformat_list(base_config_data['Rpst_true'], '_')
    return f"{time_control}-{min_elo}-{max_elo}-{n_midgame}_to_{n_endgame}-{n_boards}-P_{permute_char}-pst_{permute_pst_char}-H_{PA_KS_PS}-{permute_how_many}-{move_function}-{RP_start}-{RP_true}-{Rpst_start}-{Rpst_true}"

def define_permute_idxs(base_config_data):
    base_config_data['P_permute_idxs'] = np.array(char_to_idxs(base_config_data['permute_char']))
    base_config_data['pst_permute_idxs'] = np.array(char_to_idxs(base_config_data['permute_pst_char']))
    base_config_data['include_PA_KS_PS'] = np.array(base_config_data['include_PA_KS_PS'], dtype=bool)
    base_config_data['H_permute_idxs'] = np.array([i for i, permute_heuristic in enumerate(base_config_data['permute_H']) 
                                                   if base_config_data['include_PA_KS_PS'][i] and permute_heuristic])
    return base_config_data

def create_result_path(base_config_data, model_config_data, model_result_string, path_result=None,
                       copy_configs_flag=True):
    path = path_result if path_result is not None else join(os.getcwd(), 'results', base_config_data['model'])
    out_path = join(path,
                    f"{base_result_string(base_config_data)}---{model_result_string(model_config_data)}")

    os.makedirs(out_path, exist_ok=True)
    if copy_configs_flag: copy_configs(out_path, model_name=base_config_data['model'])
    return out_path

def load_Rs(config_data):
    RP = np.array(config_data['RP_start'], dtype=float)
    assert len(RP) == 6, f"RP must be of length 6 but was {RP}"
    Rpst = np.array(config_data['Rpst_start'], dtype=float)
    assert len(Rpst) == 6, f"Rpst must be of length 6 but was {Rpst}"
    RH = np.array(config_data['RH_start'], dtype=float) 
    assert len(RH) == 3, f"RH must be of length 3 but was {RH}"
    RH[np.invert(config_data['include_PA_KS_PS'])] = 0 # Set the values to 0 if they are not included in the model.
    return RP, Rpst, RH


# ================= Loading chess games =================
def get_boards_between(game, n_start, n_end, board_dict=None, move_dict=None, time_left_range=None):
    time_left_range = (30, np.inf) if time_left_range is None else time_left_range
    boards, moves = [], []
    board = chess.Board()
    try:
        var = game.variations[0]
    except IndexError:
        return boards, moves
    has_time = True
    for i, move in enumerate(game.mainline_moves()):
        # Search for the pattern in the input string
        match = re.search(pattern_time, var.comment)
        if not match:
            if has_time:
                print(f'No move-time information available: {var.comment} ignoring time requirement')
            has_time = False
            valid_time = True
        else:
            has_time = True
            hh = int(match.group(1))
            mm = int(match.group(2))
            ss = int(match.group(3))
            seconds_left = hh * 3600 + mm * 60 + ss
            valid_time = (time_left_range[0] < seconds_left) & (seconds_left < time_left_range[1])

        if valid_time and (n_start <= i <= n_end):
            flip = not board.turn
            boards.append(deepcopy(board))
            moves.append(move)
            if board_dict is not None and move_dict is not None:
                board_dict[i].append(deepcopy(board))
                move_dict[i].append(move)
        elif n_end < i:
            break

        var = var.variations[0] if var.variations else None
        board.push(move)

    return boards, moves


def is_valid_game(game, config_data, time_bounds=None):
    time_bounds = (180, np.inf) if time_bounds is None else time_bounds
    # Add time control check
    try:
        elo_check_white = config_data['min_elo'] < int(game.headers['WhiteElo']) < config_data['max_elo']
        elo_check_black = config_data['min_elo'] < int(game.headers['BlackElo']) < config_data['max_elo']
        # Add 1 to length check to ensure there is a valid move in the position returned
        # length_check = len(list(game.mainline_moves())) > config_data['n_endgame'] + 1
        game_type = float(game.headers['TimeControl'].split('+')[0])
        game_type_check = (time_bounds[0] < game_type) & (game_type < time_bounds[1])
        return elo_check_white and elo_check_black and game_type_check # and length_check
    except KeyError:
        return False
    except ValueError:
        return False


def get_states(websites_filepath, file_path_data, config_data, out_path, use_ply_range=True, pgn_paths=None, return_games=False, verbose=True, time_left_range=None, game_time_bounds=None):
    if 'move_percentage_data' in config_data and config_data['move_percentage_data']:
        with open('data/move_percentages/moves_1000-1200_fixed', 'r') as f:
            moves_dict = json.load(f)
        n_boards = config_data['n_boards']
        assert n_boards <= len(moves_dict), 'Number of boards exceeds number of positions.'
        sunfish_boards = [board2sunfish(fen, 0) for fen in list(moves_dict.keys())[:n_boards]]
        move_dicts = [move_dict for fen, move_dict in list(moves_dict.items())[:n_boards]]
        player_moves = [max(move_dict, key=lambda k: move_dict[k][0] - (k == 'sum')) for move_dict in move_dicts]
        player_moves = [str_to_sunfish_move(move, False) for move in player_moves]
        return sunfish_boards, player_moves

    ply_dict_boards = defaultdict(lambda: []) if use_ply_range else None
    ply_dict_moves = defaultdict(lambda: []) if use_ply_range else None

    pgn_paths = download_lichess_pgn(websites_filepath=websites_filepath,
                                     file_path_data=file_path_data,
                                     overwrite=config_data['overwrite'],
                                     n_files=config_data['n_files']) if pgn_paths is None else pgn_paths
    data_save_path = join(out_path, 'boards_and_moves.pkl')
    if use_ply_range:
        config_data = config_data
        pickle_path = f'data/processed/'
        os.makedirs(pickle_path, exist_ok=True)
        filename_unique = pgn_paths[0][-11:-4] + f'{config_data["min_elo"]}_{config_data["max_elo"]}_{config_data["n_midgame"]}_{config_data["n_endgame"]}_{config_data["n_boards"]}.pkl'
        try:
            with open(join(pickle_path, f'chess_boards_' + filename_unique), 'rb') as file:
                ply_dict_boards = pickle.load(file)
            with open(join(pickle_path, f'player_moves_' + filename_unique), 'rb') as file1:
                ply_dict_moves = pickle.load(file1)
            return ply_dict_boards, ply_dict_moves
        except FileNotFoundError:
            pass
    elif not return_games:
        try:
            with open(data_save_path, 'rb') as file:
                print(f'Found saved data at {data_save_path}')
                boards, moves = pickle.load(file)
                return boards, moves
        except FileNotFoundError:
            print(f'No saved data at {data_save_path}')
            pass

    chess_boards, moves, games = [], [], []
    i = 0
    n_games = 0
    while len(chess_boards) < config_data['n_boards']:
        pgn_path = pgn_paths[i]
        print(pgn_path)
        progress = 0
        last_len = 0
        with open(pgn_path) as pgn:
            size = os.path.getsize(pgn_path)
            with tqdm(total=size, desc=f'Looking through file {i}') as pbar:
                while len(chess_boards) < config_data['n_boards']:
                    game = chess.pgn.read_game(pgn)
                    if is_valid_game(game, config_data=config_data, time_bounds=game_time_bounds):
                        boards_, moves_ = get_boards_between(game,
                                                             config_data['n_midgame'],
                                                             config_data['n_endgame'],
                                                             board_dict=ply_dict_boards,
                                                             move_dict=ply_dict_moves,
                                                             time_left_range=time_left_range)
                        if boards_:
                            games.append(game)
                        chess_boards += boards_
                        moves += moves_

                    pbar.update(pgn.tell() - progress)
                    progress = pgn.tell()
                    if len(chess_boards) > last_len and verbose:
                        print(f'Found {len(chess_boards)}/{config_data["n_boards"]} boards so far from {n_games} games')
                        last_len = len(chess_boards)
                        n_games += 1
                    if size <= progress:
                        break
            i += 1
    if return_games:
        return games
    boards = chess_boards
    config_data['n_boards'] = min(len(boards), config_data['n_boards'])
    print(f'Using {config_data["n_boards"]} boards')
    boards, moves = boards[:config_data['n_boards']], moves[:config_data['n_boards']]
    with open(data_save_path, 'wb') as file:
        pickle.dump((boards, moves), file)

    if use_ply_range:
        with open(join(pickle_path, f'chess_boards_' + filename_unique), 'wb') as file:
            pickle.dump(dict(ply_dict_boards), file)
        with open(join(pickle_path, f'player_moves_' + filename_unique), 'wb') as file1:
            pickle.dump(dict(ply_dict_moves), file1)
        return ply_dict_boards, ply_dict_moves

    return boards, moves

# ================= Loading Previous Results =================
def load_previous_results(out_path, epoch = 0):
    _epoch = deepcopy(epoch)
    RPs, Rpsts, RHs= [], [], []
    while True:
        RP_loaded, Rpst_loaded, RH_loaded = (load_weights_epoch(out_path, result, _epoch) for result in ['Result', 'RpstResult', 'RHResult'])
        # I have to check like this because they are arrays of ambigous truth type.
        if RP_loaded is not None or Rpst_loaded is not None or RH_loaded is not None: 
            
            # Use loaded values if available, otherwise keep current values
            RP = RP_loaded if RP_loaded is not None else None
            Rpst = Rpst_loaded if Rpst_loaded is not None else None
            RH = RH_loaded if RH_loaded is not None else None

            
            RPs.append(RP)
            Rpsts.append(Rpst)
            RHs.append(RH)
            
            print(f'Results loaded for epoch {_epoch + 1}, continuing')
            assert RP is not None and Rpst is not None and RH is not None, 'All weights must be loaded, even if they are 0'
            assert _epoch < 1000
            _epoch += 1
        else:
            break
    if _epoch > epoch: print(f"Results loaded for {_epoch+1} first epochs.")
    return RPs, Rpsts, RHs, _epoch # Return the number of the epoch without data.

def load_accuracies(out_path, epoch = 0):
    _epoch = deepcopy(epoch)
    accuracies = []
    while True:
        (best_acc, temp_acc) = (load_weights_epoch(out_path, result, _epoch) for result in ['best_acc', 'temp_acc'])
        if best_acc is not None or temp_acc is not None:

            # Use loaded values if available, otherwise keep current values
            best_accuracy = best_acc if best_acc is not None else None
            temp_accuracy = temp_acc if temp_acc is not None else None
            accuracies.append((best_acc, temp_acc))
            # print(f'Accuracy loaded for epoch {epoch + 1}, continuing')
            _epoch += 1
        else:
            break
    if _epoch > epoch: print(f"Accuracies loaded for {_epoch+1} first epochs.")
    return accuracies, _epoch

# ================= Processing results =================
        
def process_epoch(RP, Rpst, RH, epoch, config_data, out_path, **kwargs):
    debug = kwargs.get('debug', False)
    csv_path = join(out_path, 'weights', f'{epoch}.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Ensure RP and Rpst have the same length
    assert len(RP) == len(Rpst), f"Length of RP and Rpst do not match: {len(RP)} != {len(Rpst)}"
    # Pad RH to match the length of RP
    RH_padded = np.pad(RH, (0, len(RP) - len(RH)), 'constant', constant_values=np.nan) # RH only contains 3 value
    # Ensure that all written arrays have the same length
    assert len(RP) == len(Rpst) == len(RH_padded) == 6, \
        f"Length of RP, Rpst and RH_padded do not match: {len(RP)}, {len(Rpst)}, {len(RH_padded)}"
    # Convert arrays to float. I can't stand more errors. 
    RP = RP.astype(float)
    Rpst = Rpst.astype(float)
    RH_padded = RH_padded.astype(float)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Result': RP,
        'RpstResult': Rpst,
        'RHResult': RH_padded,
    })

    # Write to CSV
    df.to_csv(csv_path, index=False)

    if debug:
        # Print out lengths and types for debugging
        print(f"Length of RP: {len(RP)}, Type: {type(RP)}")
        print(f"Length of Rpst: {len(Rpst)}, Type: {type(Rpst)}")
        print(f"Length of RH: {len(RH)}, Type: {type(RH)}")
        print(f"Length of RH_padded: {len(RH_padded)}, Type: {type(RH_padded)}")
        # Read back the CSV to ensure consistency
        df_check = pd.read_csv(csv_path)
        print(f"CSV contents after writing:\n{df_check}")

    if epoch and config_data['plot_every'] and (epoch + 1) % config_data['plot_every'] == 0:
        plot_R_weights(config_data=config_data,
                       out_path=out_path,
                       epoch=epoch,
                       kwargs=kwargs)
        
def save_array(array, name, out_path):

    csv_path = join(out_path, 'weights', f'{name}.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Create DataFrame
    df = pd.DataFrame({
        name: np.array(array, dtype=float),
    })
    # Write to CSV
    df.to_csv(csv_path, index=False)    

    


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
