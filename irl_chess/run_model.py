import os
import json
from os.path import join
import chess.pgn
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos, str_to_sunfish_move
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, create_result_path, get_states

if __name__ == '__main__':
    fix_cwd()
    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)

    match config_data['model']: # Load the model specified in the "base_config" file. Make sure the "model" field is set 
                                # correctly and that a model_result_string function is defined to properly store the results.
        case "sunfish_GRW": # Sunfish Greedy Random Walk
            from irl_chess.models.sunfish_GRW import run_sunfish_GRW as model, \
                                          sunfish_native_result_string as model_result_string
        case "bayesian_optimisation":
            from irl_chess.models.bayesian_optimisation import run_bayesian_optimisation as model, \
                                          bayesian_model_result_string as model_result_string
        case _:
            raise Exception(f"No model found with the name {config_data['model']}")

    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)

    if config_data['move_percentage_data']:
        print('Collecting move percentage data...')
        with open('data/move_percentages/moves_1000-1200_fixed', 'r') as f:
            moves_dict = json.load(f)
        n_boards = config_data['n_boards']
        sunfish_boards = [board2sunfish(fen, 0) for fen in list(moves_dict.keys())[:n_boards]]
        move_dicts = [move_dict for fen, move_dict in list(moves_dict.items())[:n_boards]]
        player_moves = [max(move_dict, key=lambda k: move_dict[k][0] - (k == 'sum')) for move_dict in move_dicts]
        player_moves = [str_to_sunfish_move(move, False) for move in player_moves]

    else:
        websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
        file_path_data = join(os.getcwd(), 'data', 'raw')

        sunfish_boards, player_moves = get_states(websites_filepath=websites_filepath,
                                    file_path_data=file_path_data,
                                    config_data=config_data) # Boards in the sunfish format.


    model(sunfish_boards=sunfish_boards, player_moves=player_moves, config_data=config_data, out_path=out_path)
