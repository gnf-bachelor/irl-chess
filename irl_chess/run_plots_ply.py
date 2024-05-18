import os
import json
from os.path import join
from time import time, sleep

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from irl_chess.chess_utils import sunfish_move_to_str
from irl_chess.stat_tools import wilson_score_interval


def random_moves_acc(boards, moves):
    acc, norm = 0, 0
    for board, move in tqdm(zip(boards, moves), desc='Random Moves', total=len(boards)):
        valid_moves = [el for el in board.legal_moves]
        random_move = str(np.random.choice(valid_moves))
        acc += move == random_move
        norm += 1
    return acc / norm, wilson_score_interval(acc, norm)


def result_path(config, move_min, val_prop, run_sunfish):
    return f"results/plots/maia_per_ply_{config['min_elo']}-{config['max_elo']}-{config['maia_elo']}-{config['n_boards']}-{move_min}-{val_prop}-{run_sunfish}"


def run_comparison(run_sunfish=False, pgn_paths=None, move_range=(10, 200), val_proportion=0.2):
    # SHOULD ONLY USE ALREADY TRAINED SUNFISH FOR N_BOARDS TO MAKE SENSE!!!
    from irl_chess import run_sunfish_GRW, sunfish_native_result_string, run_maia_pre, maia_pre_result_string, \
        load_config, val_sunfish_GRW,\
        union_dicts, create_result_path, get_states, board2sunfish, eval_pos

    val_proportion = val_proportion if run_sunfish else 1
    base_config_data, m_config_data = load_config()

    base_config_data['n_midgame'] = move_range[0]
    base_config_data['n_endgame'] = move_range[1]
    config_data_sunfish = union_dicts(base_config_data, m_config_data)
    with open('experiment_configs/maia_pretrained/config.json') as json_file:
        config_data_maia = json.load(json_file)
        base_config_data = union_dicts(config_data_maia, base_config_data)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    chess_boards_dict, player_moves_dict = get_states(
        websites_filepath=websites_filepath,
        file_path_data=file_path_data,
        config_data=base_config_data,
        use_ply_range=True,
        pgn_paths=pgn_paths,
        out_path=create_result_path(base_config_data, m_config_data, sunfish_native_result_string)
    )  # Boards in the sunfish format.

    acc_sunfish_list = []
    acc_maia_lists = []
    acc_random_list = []
    lower_bound_sunfish_list = []
    upper_bound_sunfish_list = []
    lower_bound_maia_list = []
    upper_bound_maia_list = []
    lower_bound_random_list = []
    upper_bound_random_list = []

    maia_model = None
    for n_moves in tqdm(range(move_range[0], move_range[1]), desc='Iterating over n_moves'):
        t = time()
        base_config_data['n_midgame'] = n_moves
        base_config_data['n_endgame'] = n_moves
        config_data_maia['n_midgame'] = n_moves
        config_data_maia['n_endgame'] = n_moves
        try:
            chess_boards = chess_boards_dict[n_moves]
            player_moves = player_moves_dict[n_moves]
        except KeyError:
            break
        out_path_sunfish = create_result_path(base_config_data, m_config_data, sunfish_native_result_string,
                                              path_result=None)
        base_config_data['model'] = 'maia_pretrained'
        out_path_maia = create_result_path(base_config_data, config_data_maia, maia_pre_result_string,
                                           path_result=None) + f'/n_moves_{n_moves}'

        val_index = int(val_proportion * len(chess_boards))
        chess_boards_train = chess_boards#[val_index:]
        chess_boards_test = chess_boards#[:val_index]
        player_moves_train = player_moves#[val_index:]
        player_moves_test = player_moves#[:val_index]

        if len(chess_boards_test) == 0:
            print(f'Only move up to {n_moves}')
            break

        validation_set = list(zip(chess_boards_test, player_moves_test))
        acc_random, (lower_bound_random, upper_bound_random) = random_moves_acc(chess_boards_test, player_moves_test)
        acc_sunfish, (lower_bound_sunfish, upper_bound_sunfish) = run_sunfish_GRW(
            chess_boards=chess_boards_train,
            player_moves=player_moves_train,
            config_data=config_data_sunfish,
            out_path=out_path_sunfish,
            validation_set=validation_set
        ) if run_sunfish else (None, (None, None))
        acc_maia, maia_model, (lower_bound_maia, upper_bound_maia) = run_maia_pre(
            chess_boards=chess_boards_train,
            player_moves=player_moves_train,
            config_data=config_data_maia,
            out_path=out_path_maia,
            validation_set=validation_set,
            model=maia_model,
            return_model=True
        )

        acc_random_list.append(acc_random)
        acc_sunfish_list.append(acc_sunfish)
        acc_maia_lists.append(acc_maia)
        lower_bound_sunfish_list.append(lower_bound_sunfish)
        upper_bound_sunfish_list.append(upper_bound_sunfish)
        lower_bound_maia_list.append(lower_bound_maia)
        upper_bound_maia_list.append(upper_bound_maia)
        lower_bound_random_list.append(lower_bound_random)
        upper_bound_random_list.append(upper_bound_random)

        plot_path = result_path(base_config_data, move_range[0], val_proportion, run_sunfish)
        csv_path = join(plot_path, f'csvs/')
        os.makedirs(csv_path, exist_ok=True)
        df_out = pd.DataFrame(
            np.array(
                (acc_random_list,
                 lower_bound_random_list,
                 upper_bound_random_list,
                 acc_sunfish_list,
                 acc_maia_lists,
                 lower_bound_maia_list,
                 upper_bound_maia_list)).T,
            columns=['random', 'random_lower', 'random_upper', 'sunfish', 'maia', 'maia_lower', 'maia_upper', ])
        df_out.to_csv(join(csv_path, f'results.csv'))

        if run_sunfish:
            plt.plot(range(move_range[0], n_moves + 1), acc_sunfish_list, label='Sunfish GRW Accuracy', color='r')
            plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_sunfish_list, upper_bound_sunfish_list,
                             alpha=0.2, color='r', label='Wilson CI')

        plt.plot(range(move_range[0], n_moves + 1), acc_maia_lists,
                 label=f'Maia {config_data_maia["maia_elo"]} Accuracy', color='b')
        plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_maia_list, upper_bound_maia_list,
                         alpha=0.2, color='b', label='Wilson CI')

        plt.plot(range(move_range[0], n_moves + 1), acc_random_list, label='Random Accuracy', color='y')
        plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_random_list, upper_bound_random_list,
                         alpha=0.2, color='y', label='Wilson CI')

        plt.title(
            f'Sunfish Accuracy vs Maia {config_data_maia["maia_elo"]} \nAccuracy from {move_range[0]} to {n_moves} moves into a game')
        plt.xlabel('Number of moves into game')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.savefig(f'{plot_path}/{n_moves}.png')
        while time() - t < 1:
            sleep(.1)
        plt.show()
        plt.close()


if __name__ == '__main__':
    pgn_paths = ['data/raw/lichess_db_standard_rated_2017-11.pgn']
    ply_range = (10, 100)

    run_comparison(run_sunfish=True, move_range=ply_range, pgn_paths=pgn_paths)
