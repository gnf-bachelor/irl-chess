import os
from os.path import join
import pandas as pd
import copy
import time
import json
from joblib import Parallel, delayed
from tqdm import tqdm
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish_move_to_str, str_to_sunfish_move, get_new_pst
from irl_chess.chess_utils.sunfish import piece, pst
from irl_chess.models.sunfish_GRW import sunfish_move
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, get_board_after_n, get_states

def load_maia_test_data(min_elo, n_boards):
    df_path = f'data/processed/maia_test/{min_elo}_{min_elo + 100}_{n_boards}.csv'
    df_path = os.path.join(*df_path.split('/'))
    dirname = os.path.dirname(df_path)
    if os.path.exists(dirname):
        for filename in os.listdir(dirname):
            if filename.endswith('.csv'):
                min_, max_, n_boards_ = [int(el) for el in filename[:-4].split('_')]
                if min_ == min_elo and n_boards <= n_boards_:
                    val_df = pd.read_csv(join(dirname, filename))
                    return val_df[:n_boards]

    return "File doesn't exist"

def correct_at_depth(player_move, state, depth):
    move = sunfish_move(state, pst, time_limit=200, max_depth=depth, move_only=True)
    return player_move == sunfish_move_to_str(move)

if __name__ == '__main__':
    while os.getcwd()[-9:] != 'irl-chess':
        os.chdir('../')

    file_path = os.path.join('experiment_configs', 'depths', 'config.json')
    with open(file_path, 'r') as file:
        config = json.load(file)

    elos = config['elos']
    n_boards = config['n_boards']

    config_data_lower = copy.copy(config) | {'min_elo': elos[0], 'max_elo': elos[0] + 100}
    config_data_upper = copy.copy(config) | {'min_elo': elos[1], 'max_elo': elos[1] + 100}

    if config['data'] == 'maia':
        val_df_lower = load_maia_test_data(elos[0], n_boards)
        sunfish_boards_lower = val_df_lower['board'].apply(lambda x: board2sunfish(x, 0))
        player_moves_lower = val_df_lower['move']

        val_df_upper = load_maia_test_data(elos[1], n_boards)
        sunfish_boards_upper = val_df_upper['board'].apply(lambda x: board2sunfish(x, 0))
        player_moves_upper = val_df_upper['move']

    elif config['data'] == 'lichess':
        websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
        file_path_data = join(os.getcwd(), 'data', 'raw')

        sunfish_boards_lower, player_moves_lower = get_states(websites_filepath=websites_filepath,
                                                  file_path_data=file_path_data,
                                                  config_data=config_data_lower)  # Boards in the sunfish format.

        sunfish_boards_upper, player_moves_upper = get_states(websites_filepath=websites_filepath,
                                                  file_path_data=file_path_data,
                                                  config_data=config_data_upper)  # Boards in the sunfish format.

        player_moves_lower = [sunfish_move_to_str(move) for move in player_moves_lower]
        player_moves_upper = [sunfish_move_to_str(move) for move in player_moves_upper]


    n_threads = config['n_threads']
    with Parallel(n_jobs=n_threads) as parallel:
        depths = config['depths']
        depth_scores_lower = [0] * len(depths)
        depth_scores_upper = [0] * len(depths)
        for i, depth in enumerate(depths):
            print(f'\n Depth: {depth} \n', '-'*20)
            correct_lower = parallel(delayed(correct_at_depth)(player_move, state, depth)
                                     for player_move, state in tqdm(list(zip(player_moves_lower, sunfish_boards_lower))))
            correct_upper = parallel(delayed(correct_at_depth)(player_move, state, depth)
                                     for player_move, state in tqdm(list(zip(player_moves_upper, sunfish_boards_upper))))
            depth_scores_lower[i] = sum(correct_lower)/n_boards
            depth_scores_upper[i] = sum(correct_upper)/n_boards
    lower_min_elo = config_data_lower['min_elo']
    upper_min_elo = config_data_upper['min_elo']
    print(f'Depth scores for ELO {lower_min_elo}-{lower_min_elo + 100}: {depth_scores_lower}')
    print(f'Depth scores for ELO {upper_min_elo}-{upper_min_elo + 100}: {depth_scores_upper}')