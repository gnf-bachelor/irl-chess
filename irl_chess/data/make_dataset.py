import json
import re
import os
import time
import requests
import pyzstd

import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join


def decompress_zstd(zstd_path, extract_path):
    """
    Given a path to a .zst file, decompress the .zst
    and extract the files within it, then delete the
    .zst file
    :param zstd_path:
    :param extract_path:
    :param overwrite:
    :return:
    """
    with open(zstd_path, 'rb') as zstd_file:
        compressed_data = zstd_file.read()
        decompressed_data = pyzstd.decompress(compressed_data)

    destination_path = extract_path
    print(f'Attempting to decompress to: {destination_path}')
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    with open(destination_path, 'wb') as decompressed_file:
        decompressed_file.write(decompressed_data)
    os.remove(zstd_path)
    print(f"Decompressed: {zstd_path} and deleted the zip file!")



def download_file(url, destination):
    """
    Use the requests package to download an url and save
    the file to the given destination. Returns True if
    the file was successfully downloaded else False.
    :param url:
    :param destination:
    :return:
    """
    try:
        response = requests.get(url)
        with open(destination, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded: {url}")

        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def parse_moves(moves):
    """
    Parse the moves of a chess game from the Lichess chess game format.
    :param moves:
    :return:
    """
    out = ''
    for el in moves.split(' '):
        if '.' not in el:
            out += el + ','
    return [out]


def parse_game(input_string, included_keys):
    pattern = r'\[([^\]]+)\]'

    # Use regular expression to find all matches within square brackets
    matches = re.findall(pattern, input_string)

    # Specify the keys you want to include
    # included_keys = ['Event', 'Site', 'Date', 'White', 'Black', 'Result', 'UTCDate',
    #                  'UTCTime', 'WhiteElo', 'BlackElo', 'WhiteRatingDiff', 'BlackRatingDiff',
    #                  'WhiteTitle', 'ECO', 'Opening', 'TimeControl', 'Termination']

    # Create a list to store the extracted values
    data_list = []
    for match in matches:
        key, value = map(str.strip, match.split(' ', 1))

        # Include only the specified keys
        if key.lower() == 'timecontrol':
            data_list += value.split('+')
        elif key in included_keys:
            # Remove double quotes from the value
            value = value.replace('"', '')
            data_list.append(value)

    return data_list


def txt_to_csv(filename, overwrite=True):
    """
    Given a .pgn file from the lichess database, convert it to a .csv file
    and return the path. There is something funky going on with pandas
    where it creates an extra column full of None values. If a row has an
    invalid length it is ignored.
    :param filename:
    :param overwrite:
    :return:
    """
    filename_out = filename[:-4].replace('raw', 'processed') + '.csv'
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    if not overwrite and os.path.exists(filename_out):
        print(f'{filename_out} already exists and was not changed')
        return filename_out

    columns = ['Event', 'Site', 'White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo',
               'WhiteRatingDiff', 'BlackRatingDiff', 'ECO', 'Opening', 'TimeStart', 'TimeIncrement',
               'Termination', 'Moves']
    data_raw = []
    with open(filename, 'r') as f:
        game = ''
        for line in tqdm(f.readlines(), 'Converting to csv'):
            if line[0] == '1' or line == ' 0-1\n':
                row = parse_game(game, included_keys=columns[:-1] + ['TimeControl']) + parse_moves(line)
                if len(row) == len(columns):    # Only pass valid rows... sadly necessary
                    data_raw.append(row)
                game = ''
            elif line.strip():
                game += line

        df = pd.DataFrame(data_raw, columns=columns)
        df.dropna(inplace=True)
        df.to_csv(filename_out, index=False, mode='w')
    print(f'Converted .txt to .csv!')
    return filename_out


def download_lichess_pgn(websites_filepath, file_path_data, n_files=np.inf, overwrite=True):
    """
    Given a list of websites and the path to the data folders
    download the data from the websites using the helpers defined
    above. This function is always verbose and returns a list of
    the paths to the .csv files in question.
    :param websites_filepath:
    :param file_path_data:
    :param n_files:
    :param overwrite:
    :return:
    """
    os.makedirs(file_path_data, exist_ok=True)
    start = time.time()
    filepaths_out = []
    with open(websites_filepath, 'r') as filename:
        urls = filename.readlines()
        urls = [url.strip() for url in urls]

    for i, url in enumerate(urls, start=1):
        start_file = time.time()
        destination = join(file_path_data, url.split("/")[-1])
        print(f'\n\n-------------------  {i}/{len(urls)}  -------------------\n\n')
        filepath_txt = destination[:-4]
        if overwrite or not os.path.exists(filepath_txt):
            os.makedirs(os.path.dirname(filepath_txt), exist_ok=True)
            if download_file(url, destination):
                decompress_zstd(destination, extract_path=filepath_txt)
        # filepath_csv = txt_to_csv(filepath_txt, overwrite=overwrite)
        filepaths_out.append(filepath_txt)
        print(f'Time taken: {time.time() - start_file:.2f} seconds for file')
        print(f'Time taken: {time.time() - start:.2f} seconds in total')

        if i == n_files:
            break

    return filepaths_out


def load_lichess_dfs(websites_filepath, file_path_data, n_files, overwrite=False):
    """
    Given the desired year and month as strings, load the lichess
    data from the csv file and return it as a DataFrame.
    Month must have 2 digits.
    No errors are caught in the function.
    :param year:    YYYY
    :param month:   MM
    :return:
    """
    datapaths = download_lichess_pgn(websites_filepath, file_path_data, n_files=n_files, overwrite=overwrite)
    df = pd.read_csv(datapaths[0], index_col=None)
    for path in tqdm(datapaths[1:], desc='Contatenating DataFrames'):
        df_ = pd.read_csv(path, index_col=None)
        df = pd.concat((df, df_), axis=0)
    df.dropna(inplace=True)
    return df


def make_maia_csv(filepath, n_games):
    # File too big for python unzip to work, so must be manually downloaded and unzipped
    pass


if __name__ == "__main__":
    from irl_chess import get_states

    pgn_paths = ['data/raw/lichess_db_standard_rated_2017-11.pgn']
    csv_path = 'data/processed'
    ply_range = (10, 200)
    os.makedirs(csv_path, exist_ok=True)

    with open('experiment_configs/base_config.json', 'r') as file:
        config = json.load(file)

    config['n_midgame'], config['n_endgame'] = ply_range
    chess_board_dict, player_move_dict = get_states(None, None, config, pgn_paths=pgn_paths)

    for ply in range(*ply_range):
        chess_boards = chess_board_dict[ply]

