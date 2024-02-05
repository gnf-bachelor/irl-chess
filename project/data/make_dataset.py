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
    try:
        with open(zstd_path, 'rb') as zstd_file:
            compressed_data = zstd_file.read()
            decompressed_data = pyzstd.decompress(compressed_data)

        destination_path = extract_path

        with open(destination_path, 'wb') as decompressed_file:
            decompressed_file.write(decompressed_data)

        print(f"Decompressed: {zstd_path}")
        os.remove(zstd_path)
        print(f'Unzipped and deleted the zip file!')

    except Exception as e:
        print(f"Failed to decompress {zstd_path}: {e}")


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
    return out


def txt_to_csv(filename, overwrite=True):
    """
    Given a .pgn file from the lichess database, convert it to a .csv file
    and return the path. There is something funky going on with pandas
    where it creates an extra column full of None values.
    :param filename:
    :param overwrite:
    :return:
    """
    filename_out = filename[:-4].replace('raw', 'processed') + '.csv'

    if not overwrite:
        try:
            df = pd.read_csv(filename_out, index_col=0)
            print(f'{filename_out} already exists and was not changed')
            return filename_out
        except FileNotFoundError:
            pass

    columns = ['Event', 'Site', 'White', 'Black', 'Result', 'UTCDate', 'UTCTime', 'WhiteElo', 'BlackElo',
               'WhiteRatingDiff', 'BlackRatingDiff', 'ECO', 'Opening', 'TimeStart', 'TimeIncrement', 'Termination', 'Moves']
    data_raw = []
    with open(filename, 'r') as f:
        game = []
        for line in tqdm(f.readlines(), 'Converting to csv'):
            if line[0] == '1':
                game.append(parse_moves(line.strip()))
                data_raw.append(game)
                game = []
            elif line.strip():
                for el in line.split('"')[-2].split('+'):
                    if el.strip():
                        game.append(el)
    # For some reason pandas seems to add a column of None values...
    df = pd.DataFrame(data_raw, columns=columns + ['None'])
    df = df.iloc[:, :-1]
    df.to_csv(filename_out, index=False)
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
    start = time.time()
    filepaths_csv = []
    try:
        with open(websites_filepath, 'r') as filename:
            urls = filename.readlines()
            urls = [url.strip() for url in urls]

        for i, url in enumerate(urls, start=1):
            start_file = time.time()
            destination = join(file_path_data, url.split("/")[-1])
            print(f'\n\n-------------------  {i}/{len(urls)}  -------------------\n\n')
            filepath_out = destination[:-4]
            if overwrite or not os.path.exists(filepath_out):
                if download_file(url, destination):
                    decompress_zstd(destination, extract_path=filepath_out)
            filepath_csv = txt_to_csv(filepath_out, overwrite=overwrite)
            filepaths_csv.append(filepath_csv)
            print(f'Time taken: {time.time() - start_file:.2f} seconds for file')
            print(f'Time taken: {time.time() - start:.2f} seconds in total')

            if i == n_files:
                break

    except FileNotFoundError:
        print(f"File not found: {websites_filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return filepaths_csv


def load_lichess_csv(year, month):
    """
    Given the desired year and month as strings, load the lichess
    data from the csv file and return it as a DataFrame.
    Month must have 2 digits.
    No errors are caught in the function.
    :param year:    YYYY
    :param month:   MM
    :return:
    """
    file_path_data = join(os.getcwd(), 'data', 'processed')
    file = join(file_path_data, f'lichess_db_standard_rated_{year}-{month}.csv')
    return pd.read_csv(file, index_col=None)


if __name__ == "__main__":
    n_files = np.inf
    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    download_lichess_pgn(websites_filepath, file_path_data, n_files=n_files, overwrite=False)
