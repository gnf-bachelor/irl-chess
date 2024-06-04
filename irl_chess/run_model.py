import os
from os.path import join
import chess.pgn
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos, str_to_sunfish_move
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, create_result_path, get_states

if __name__ == '__main__':
    fix_cwd()
    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)

    base_config_data['overwrite'] = False
    base_config_data['n_files'] = 6
    base_config_data['time_control'] = False
    base_config_data['min_elo'] = 1900
    base_config_data['max_elo'] = 2000
    base_config_data['n_midgame'] = 15
    base_config_data['n_endgame'] = 30
    base_config_data['n_boards'] = 10000
    base_config_data['epochs'] = 400
    base_config_data['n_threads'] = 24
    base_config_data['plot_every'] = 50
    base_config_data['save_every'] = 50

    base_config_data['plot_char'] = ["P", "N", "B", "R", "Q"]
    base_config_data['permute_char'] = ["N", "B", "R", "Q"]
    base_config_data['R_start'] = [100, 100, 100, 100, 100, 60000]

    base_config_data['move_function'] = "sunfish_move"
    base_config_data['R_true'] = [100, 280, 320, 480, 920, 60000]
    base_config_data['model'] = "sunfish_GRW"
    base_config_data['move_percentage_data'] = False

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
    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards, player_moves = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data) # Boards in the sunfish format.

    model(sunfish_boards=sunfish_boards, player_moves=player_moves, config_data=config_data, out_path=out_path)
