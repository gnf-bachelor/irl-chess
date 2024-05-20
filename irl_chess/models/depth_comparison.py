import os
from os.path import join
import chess.pgn
import numpy as np
import json
import requests
from bs4 import BeautifulSoup
from requests_html import HTMLSession
from joblib import Parallel, delayed
from tqdm import tqdm
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos, sunfish_move_to_str, str_to_sunfish_move, get_new_pst
from irl_chess.chess_utils.sunfish import piece, pst
from irl_chess.models.sunfish_GRW import sunfish_move
from collections import defaultdict
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, get_board_after_n, get_states

with open('experiment_configs\\depths\\config_lower.json', 'r') as file:
    config_lower = json.load(file)

with open('experiment_configs\\depths\\config_upper.json', 'r') as file:
    config_upper = json.load(file)

websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
file_path_data = join(os.getcwd(), 'data', 'raw')

sunfish_boards_lower, player_moves_lower = get_states(websites_filepath=websites_filepath,
                                                        file_path_data=file_path_data,
                                                        config_data=config_lower)
player_moves_lower = [sunfish_move_to_str(move) for move in player_moves_lower]

sunfish_boards_upper, player_moves_upper = get_states(websites_filepath=websites_filepath,
                                                        file_path_data=file_path_data,
                                                        config_data=config_upper)
player_moves_upper = [sunfish_move_to_str(move) for move in player_moves_upper]


def correct_at_depth(player_move, state, depth):
    move = sunfish_move(state, pst, time_limit=200, max_depth=depth, move_only=True)
    #top_5_moves = top_k_moves(move_dict, 5)
    return player_move == sunfish_move_to_str(move)

n_threads = config_lower['n_threads']
n_boards = config_lower['n_boards']
with Parallel(n_jobs=n_threads) as parallel:
    depths = [1,2,3]
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
lower_min_elo = config_lower['min_elo']
lower_max_elo = config_lower['max_elo']
upper_min_elo = config_upper['min_elo']
upper_max_elo = config_upper['max_elo']
print(f'Depth scores for ELO {lower_min_elo}-{lower_max_elo}: {depth_scores_lower}')
print(f'Depth scores for ELO {upper_min_elo}-{upper_max_elo}: {depth_scores_upper}')