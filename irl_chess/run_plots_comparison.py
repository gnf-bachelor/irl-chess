import os
import json
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from irl_chess.chess_utils import sunfish_move_to_str


def random_moves_acc(boards, moves):
    acc, norm = 0, 0
    for board, move in tqdm(zip(boards, moves), desc='Random Moves', total=len(boards)):
        valid_moves = [el for el in board.legal_moves]
        random_move = str(np.random.choice(valid_moves))
        acc += sunfish_move_to_str(move) == random_move
        norm += 1
    return acc / norm


def result_path(config, move_min, val_prop):
    return f"results/plots/{config['max_elo']}-{config['min_elo']}-{config['maia_elo']}-{config['n_boards']}-{move_min}-{val_prop}"


if __name__ == '__main__':
    from irl_chess import run_sunfish_GRW, sunfish_native_result_string, run_maia_pre, maia_pre_result_string, \
        load_config, \
        union_dicts, create_result_path, get_states, board2sunfish, eval_pos

    base_config_data, m_config_data = load_config()
    move_range = (10, 31)   # (inclusive, exclusive)
    val_proportion = 0.2

    base_config_data['n_endgame'] = move_range[1]
    config_data_sunfish = union_dicts(base_config_data, m_config_data)
    with open('experiment_configs/maia_pretrained/config.json') as json_file:
        config_data_maia = json.load(json_file)
        config_data_maia = union_dicts(base_config_data, config_data_maia)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    chess_boards_dict, player_moves_dict = get_states(
        websites_filepath=websites_filepath,
        file_path_data=file_path_data,
        config_data=config_data_maia,
        ply_range=move_range
    )  # Boards in the sunfish format.

    acc_sunfish_list = []
    acc_maia_list = []
    acc_random_list = []
    maia_model = None
    for n_moves in tqdm(range(move_range[0], move_range[1]), desc='Iterating over n_moves'):
        base_config_data['n_midgame'] = n_moves
        base_config_data['n_endgame'] = n_moves
        config_data_maia['n_midgame'] = n_moves
        config_data_maia['n_endgame'] = n_moves
        chess_boards = chess_boards_dict[n_moves]
        player_moves = player_moves_dict[n_moves]

        out_path_sunfish = create_result_path(base_config_data, m_config_data, sunfish_native_result_string,
                                              path_result=None)
        out_path_maia = create_result_path(base_config_data, config_data_maia, maia_pre_result_string, path_result=None)

        val_index = int(val_proportion * len(chess_boards))
        sunfish_boards_train = [board2sunfish(board, eval_pos(board)) for board in chess_boards[val_index:]]
        sunfish_boards_test = [board2sunfish(board, eval_pos(board)) for board in chess_boards[:val_index]]
        chess_boards_test = chess_boards[:val_index]
        player_moves_train = player_moves[val_index:]
        player_moves_test = player_moves[:val_index]

        validation_set_sunfish = list(zip(sunfish_boards_test, player_moves_test))
        validation_set_maia = list(zip(chess_boards_test, player_moves_test))
        acc_random = random_moves_acc(chess_boards_test, player_moves_test)
        acc_sunfish = run_sunfish_GRW(
            sunfish_boards=sunfish_boards_train,
            player_moves=player_moves_train,
            config_data=config_data_sunfish,
            out_path=out_path_sunfish,
            validation_set=validation_set_sunfish
        )
        acc_maia, maia_model = run_maia_pre(
            sunfish_boards=sunfish_boards_train,
            player_moves=player_moves_train,
            config_data=config_data_maia,
            out_path=out_path_maia,
            validation_set=validation_set_maia,
            model=maia_model,
            return_model=True
        )

        acc_random_list.append(acc_random)
        acc_sunfish_list.append(acc_sunfish)
        acc_maia_list.append(acc_maia)

        plt.plot(range(move_range[0], n_moves+1), acc_sunfish_list, label='Sunfish GRW Accuracy')
        plt.plot(range(move_range[0], n_moves+1), acc_maia_list, label=f'Maia {config_data_maia["maia_elo"]} Accuracy')
        plt.plot(range(move_range[0], n_moves+1), acc_random_list, label='Random Accuracy')
        plt.title(f'Sunfish Accuracy vs Maia {config_data_maia["maia_elo"]} Accuracy {move_range[0]} to {n_moves} number of moves into a game')
        plt.xlabel('Number of moves into game')
        plt.ylabel('Accuracy')
        plt.legend()
        plot_path = result_path(config_data_maia, move_range[0], val_proportion)
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(f'{plot_path}/{n_moves}.png')
        plt.show()
        plt.close()
