import os

import numpy as np
import opendatasets as od
import json
import pandas as pd
from tqdm import tqdm

dataset_URL = "https://www.kaggle.com/datasets/milesh1/35-million-chess-games"
data_folder = "landuse-scene-classification"


def make_dataset():
    """
    Description:
        Downloads the dataset into data and extracts the content using the opendatasets library.
        Important to have the kaggle.json file in the root folder.
    Returns:
        None

    """
    od.download(dataset_URL, data_dir="data/", unzip=True)
    try:
        os.rename('data/35-million-chess-games/all_with_filtered_anotations_since1998.txt', 'data/35-million-chess-games/all_with_filtered_annotations_since1998.txt')
        print('Fixed spelling error')
    except FileNotFoundError:
        pass


def win_loss_translation(outcome):
    # positive for white, 2 for checkmate, 1 for resignation
    if outcome[0] == '0':  # White loss
        out = -1
    elif outcome[1] == '-':  # White win
        out = 1
    else:  # Draw
        out = 0
    return out


def parse_pgn(game):
    # CHATGPT (Modified)
    white_elo_start = game.find("WhiteElo") + 10
    white_elo_end = white_elo_start + game[white_elo_start:].find('"')
    white_elo = int(game[white_elo_start:white_elo_end].strip())

    black_elo_start = game.find("BlackElo") + 10
    black_elo_end = black_elo_start + game[black_elo_start:].find('"')
    black_elo = int(game[black_elo_start:black_elo_end].strip())

    moves = []
    game_split = game.split("\n\n")[1].split(' ')
    for move in game_split:
        if '{' in move:
            break
        if '.' not in move:  # Remove numbering
            moves.append(move)
    outcome = win_loss_translation(game_split[-1])  # Get reason for game ending

    game_length = len(moves)
    # (for data), moves
    return (white_elo, black_elo, game_length, outcome), moves


def load_games(path, dtype=np.int16):
    """
    Returns a data matrix consisting of
    (White ELO, Black ELO, Game Length, Outcome)
    along with a list containing strings of the
    moves in standard algebraic notation (san)
    :param path:
    :param dtype:
    :return:
    """
    moves_list = []
    data = []
    path_data = path.replace(path.split('/')[-1], 'data.csv')
    path_moves = path.replace(path.split('/')[-1], 'moves.txt')
    try:
        data = pd.read_csv(path_data, sep=',', index_col=0).values
        moves_list = []
        with open(path_moves, 'r') as file:
            for line in file:
                moves_list.append(line.split())
        print('Stored files successfully loaded')
        return moves_list, data
    except FileNotFoundError:
        pass

    with open(path, 'r') as file:
        game = ""
        event_count = 0
        for line in tqdm(file, total=8_936_708, desc='Parsing'):    # Copied from the data file, hard to do dynamically
            game += line
            if 'Event' in line:
                event_count += 1
            if event_count == 2:  # Check if game string is not empty
                (temp), moves = parse_pgn(game)
                moves_list.append(moves)
                data.append(temp)
                game = ""
                event_count = 1
    data = np.array(data, dtype=dtype)
    pd.DataFrame(data).to_csv(path_data, sep=',')
    with open(path_moves, 'w') as file:
        for move in tqdm(moves_list, desc='Saving Moves'):
            temp_move = ''
            for char in move:
                temp_move += char + ' '
            file.writelines(temp_move[:-1] + '\n')
    print('Data loaded and files successfully created and saved')
    return moves_list, data


if __name__ == "__main__":
    make_dataset()
