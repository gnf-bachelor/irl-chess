import os
import json
from os.path import join

import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    from irl_chess import run_sunfish_GRW, sunfish_native_result_string, run_maia_pre, maia_pre_result_string, \
        load_config, \
        union_dicts, create_result_path, get_states, board2sunfish, eval_pos

    base_config_data, m_config_data = load_config()
    config_data_sunfish = union_dicts(base_config_data, m_config_data)
    with open('experiment_configs/maia_pretrained/config.json') as json_file:
        config_data_maia = json.load(json_file)
        config_data_maia = union_dicts(base_config_data, config_data_maia)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    chess_boards, player_moves = get_states(
        websites_filepath=websites_filepath,
        file_path_data=file_path_data,
        config_data=config_data_maia
    )  # Boards in the sunfish format.
    sunfish_boards = [board2sunfish(board, eval_pos(board)) for board in chess_boards]

    move_range = (12, 31)   # (inclusive, exclusive)
    acc_sunfish_list = []
    acc_maia_list = []
    maia_model = None
    for n_moves in tqdm(range(move_range[0], move_range[1]), desc='Iterating over n_moves'):
        base_config_data['n_midgame'] = n_moves
        base_config_data['n_endgame'] = n_moves
        out_path_sunfish = create_result_path(base_config_data, m_config_data, sunfish_native_result_string,
                                              path_result=None)
        out_path_maia = create_result_path(base_config_data, config_data_maia, maia_pre_result_string, path_result=None)

        validation_set_sunfish = list(zip(sunfish_boards, player_moves))
        validation_set_maia = list(zip(chess_boards, player_moves))
        acc_sunfish = run_sunfish_GRW(
            sunfish_boards=sunfish_boards,
            player_moves=player_moves,
            config_data=config_data_sunfish,
            out_path=out_path_sunfish,
            validation_set=validation_set_sunfish
        )
        acc_maia, maia_model = run_maia_pre(
            sunfish_boards=sunfish_boards,
            player_moves=player_moves,
            config_data=config_data_maia,
            out_path=out_path_maia,
            validation_set=validation_set_maia,
            model=maia_model,
            return_model=True
        )

        acc_sunfish_list.append(acc_sunfish)
        acc_maia_list.append(acc_maia)

        plt.plot(range(move_range[0], n_moves+1), acc_sunfish_list, label='Sunfish Accuracy')
        plt.plot(range(move_range[0], n_moves+1), acc_maia_list, label='Maia Accuracy')
        plt.title('Sunfish Accuracy vs Maia Accuracy by number of moves into a game')
        plt.xlabel('Number of moves into game')
        plt.ylabel('Accuracy')
        plt.legend()
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig(f'results/plots/sunfish_maia_accuracy_{move_range[0]}_{n_moves}.png')
        plt.show()
        plt.close()
