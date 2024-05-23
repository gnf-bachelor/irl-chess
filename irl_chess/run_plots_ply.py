import os
import json
from os.path import join
from time import time, sleep

import chess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from irl_chess import load_maia_test_data
from irl_chess.chess_utils import sunfish_move_to_str
from irl_chess.stat_tools import wilson_score_interval
from irl_chess.visualizations import maia_palette_name, sunfish_palette_name


def random_moves_acc(boards, moves):
    acc, norm = 0, 0
    for board, move in tqdm(zip(boards, moves), desc='Random Moves', total=len(boards)):
        valid_moves = [el for el in board.legal_moves]
        random_move = str(np.random.choice(valid_moves))
        acc += str(move) == str(random_move)
        norm += 1
    return acc / norm, wilson_score_interval(acc, norm)


def result_path(config, move_min, val_prop, run_sunfish, sunfish_epoch, using_maia_val_data):
    return f"results/plots/maia_per_ply_{config['min_elo']}-{config['max_elo']}-{config['maia_elo']}-{config['n_boards']}-{move_min}-{val_prop}-{run_sunfish}-{sunfish_epoch}-{using_maia_val_data}"


def run_comparison(run_sunfish=False, pgn_paths=None, move_range=(10, 100), val_proportion=0.2, sunfish_epoch=90,
                   using_maia_val_data=False, using_default_sunfish=False):
    # SHOULD ONLY USE ALREADY TRAINED SUNFISH FOR N_BOARDS TO MAKE SENSE!!!
    from irl_chess import run_sunfish_GRW, sunfish_native_result_string, run_maia_pre, maia_pre_result_string, \
        load_config, val_sunfish_GRW, \
        union_dicts, create_result_path, get_states, board2sunfish, eval_pos

    val_proportion = val_proportion if run_sunfish else 1
    base_config_data, m_config_data = load_config('sunfish_GRW')
    config_data_sunfish = union_dicts(base_config_data, m_config_data)

    # base_config_data['n_midgame'] = move_range[0]
    # base_config_data['n_endgame'] = move_range[1]

    with open('experiment_configs/maia_pretrained/config.json') as json_file:
        config_data_maia = json.load(json_file)
        base_config_data = union_dicts(config_data_maia, base_config_data)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')
    out_path_sunfish = create_result_path(base_config_data, m_config_data, sunfish_native_result_string)
    out_path_maia = create_result_path(base_config_data, config_data_maia, maia_pre_result_string) + '-maia'

    if using_maia_val_data:
        val_df = load_maia_test_data(base_config_data['min_elo'], base_config_data['n_boards'])
    else:
        chess_boards_dict, player_moves_dict = get_states(
            websites_filepath=websites_filepath,
            file_path_data=file_path_data,
            config_data=base_config_data,
            use_ply_range=True,
            pgn_paths=pgn_paths,
            out_path=out_path_sunfish
        )

    acc_sunfish_list = []
    acc_maia_list = []
    acc_random_list = []
    lower_bound_sunfish_list = []
    upper_bound_sunfish_list = []
    lower_bound_maia_list = []
    upper_bound_maia_list = []
    lower_bound_random_list = []
    upper_bound_random_list = []

    acc_lists = [acc_random_list,
                 lower_bound_random_list,
                 upper_bound_random_list,
                 acc_sunfish_list,
                 lower_bound_sunfish_list,
                 upper_bound_sunfish_list,
                 acc_maia_list,
                 lower_bound_maia_list,
                 upper_bound_maia_list]

    maia_model = None
    results_path = result_path(base_config_data, move_range[0], val_proportion, run_sunfish, sunfish_epoch,
                               using_maia_val_data) + ('using_default_sunfish' if using_default_sunfish else '')
    csv_path = join(results_path, f'csvs', f'results.csv')
    plot_path_base = join(results_path, f'plots', )
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(plot_path_base, exist_ok=True)

    offset = 0
    try:
        df_results = pd.read_csv(csv_path)
        offset = len(df_results)
        for i, column in enumerate(df_results.values.T):
            acc_lists[i] += list(column)
    except FileNotFoundError:
        pass

    for n_moves in tqdm(range(move_range[0] + offset, move_range[1]), desc='Iterating over n_moves'):
        plot_path = join(plot_path_base, f'{n_moves}.svg')

        t = time()

        if using_maia_val_data:
            move_mask = (n_moves <= val_df['move_ply']) & (val_df['move_ply'] <= n_moves)
            chess_boards = [chess.Board(board) for board in val_df['board'][move_mask]]
            player_moves = [move for move in val_df['move'][move_mask]]
        else:
            try:
                chess_boards = chess_boards_dict[n_moves]
                player_moves = player_moves_dict[n_moves]
            except KeyError:
                break

        if len(chess_boards) == 0:
            print(f'Only move up to {n_moves}')
            break

        validation_set = list(zip(chess_boards, player_moves))
        acc_random, (lower_bound_random, upper_bound_random) = random_moves_acc(chess_boards, player_moves)
        acc_maia, maia_model, (lower_bound_maia, upper_bound_maia) = run_maia_pre(
            chess_boards=chess_boards,
            player_moves=player_moves,
            config_data=config_data_maia,
            out_path=out_path_maia + f'/n_moves_{n_moves}',
            validation_set=validation_set,
            model=maia_model,
            return_model=True
        )
        if using_default_sunfish:
            out_path_sunfish = join(os.path.dirname(out_path_sunfish), 'default_sunfish')
            default_weight_path = join(out_path_sunfish, 'weights')
            os.makedirs(default_weight_path, exist_ok=True)
            R_default = np.array(config_data_sunfish['R_true'])
            df = pd.DataFrame(R_default.T, columns=['Result'])
            df.to_csv(join(default_weight_path, '0.csv'))
        acc_sunfish, (lower_bound_sunfish, upper_bound_sunfish) = val_sunfish_GRW(
            epoch=0 if using_default_sunfish else sunfish_epoch,
            use_player_moves=True,
            config_data=config_data_sunfish,
            out_path=out_path_sunfish,
            validation_set=validation_set,
            name=f'n_moves/{n_moves}'
            ) if run_sunfish else (None, (None, None))

        acc_random_list.append(acc_random)
        acc_sunfish_list.append(acc_sunfish)
        acc_maia_list.append(acc_maia)
        lower_bound_sunfish_list.append(lower_bound_sunfish)
        upper_bound_sunfish_list.append(upper_bound_sunfish)
        lower_bound_maia_list.append(lower_bound_maia)
        upper_bound_maia_list.append(upper_bound_maia)
        lower_bound_random_list.append(lower_bound_random)
        upper_bound_random_list.append(upper_bound_random)


        alpha = 0.1
        sunfish_palette = sns.color_palette(sunfish_palette_name, 2)
        maia_palette = sns.color_palette(maia_palette_name, 2)
        random_palette = sns.color_palette("viridis", 2)
        if run_sunfish:
            plt.plot(range(move_range[0], n_moves + 1), acc_sunfish_list,
                     label=f'Sunfish {"" if using_default_sunfish else "GRW"}\nAccuracy', color=sunfish_palette[0])
            plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_sunfish_list, upper_bound_sunfish_list,
                             alpha=alpha, color=sunfish_palette[0], label='Wilson CI')

        plt.plot(range(move_range[0], n_moves + 1), acc_maia_list,
                 label=f'Maia {config_data_maia["maia_elo"]}\nAccuracy', color=maia_palette[0])
        plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_maia_list, upper_bound_maia_list,
                         alpha=alpha, color=maia_palette[0], label='Wilson CI')

        plt.plot(range(move_range[0], n_moves + 1), acc_random_list, label='Random\nAccuracy', color=random_palette[-1])
        plt.fill_between(range(move_range[0], n_moves + 1), lower_bound_random_list, upper_bound_random_list,
                         alpha=alpha, color=random_palette[-1], label='Wilson CI')

        plt.title(
            f'Sunfish {"" if using_default_sunfish else "GRW"} Accuracy vs Maia {config_data_maia["maia_elo"]} \nAccuracy from {move_range[0]} to {n_moves} moves into a game')
        plt.xlabel('Number of moves into game')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()

        plt.savefig(plot_path)
        while time() - t < 1:
            sleep(.1)

        df_out = pd.DataFrame(
            np.array(acc_lists).T,
            columns=['random', 'random_lower', 'random_upper', 'sunfish', 'sunfish_lower', 'sunfish_upper', 'maia',
                     'maia_lower', 'maia_upper', ])
        df_out.to_csv(csv_path, index=False)

        plt.show()
        plt.close()
        print(f'Using sunfish with following config:')
        print(config_data_sunfish)


if __name__ == '__main__':
    pgn_paths = ['data/raw/lichess_db_standard_rated_2017-11.pgn']
    ply_range = (10, 81)
    # Set the param epochs in the base config to specify epochs for sunfish
    # Also remember to set the move function to player move as this is used for validation
    run_comparison(run_sunfish=True, move_range=ply_range, pgn_paths=pgn_paths, using_maia_val_data=True,
                   using_default_sunfish=True)
