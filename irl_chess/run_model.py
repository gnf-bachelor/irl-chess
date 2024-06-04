import os
from os.path import join
import chess
import logging

from irl_chess.misc_utils.utils import union_dicts
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, create_result_path, get_states
from irl_chess import load_maia_test_data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    fix_cwd()
    base_config_data, model_config_data = load_config()
    base_config_data['overwrite'] = False
    base_config_data['n_files'] = 1
    base_config_data['time_control'] = False
    base_config_data['min_elo'] = 1100
    base_config_data['max_elo'] = 1200
    base_config_data['n_midgame'] = 10
    base_config_data['n_endgame'] = 100
    base_config_data['n_boards'] = 10000
    base_config_data['n_boards_val'] = 10
    base_config_data['epochs'] = 300
    base_config_data['max_hours'] = 100
    base_config_data['n_threads'] = -1
    base_config_data['plot_every'] = 20
    base_config_data['val_every'] = 10
    base_config_data['run_n_times'] = 5

    base_config_data['plot_char'] = ["P", "N", "B", "R", "Q"]
    base_config_data['permute_char'] = []
    base_config_data['RP_start'] = [100, 280, 320, 479, 929, 60000]

    base_config_data['plot_pst_char'] = ["P", "N", "B", "R", "Q", "K"]
    base_config_data['permute_pst_char'] = ["P", "N", "B", "R", "Q", "K"]
    base_config_data['Rpst_start'] = [0, 0, 0, 0, 0, 0]

    base_config_data['include_PA_KS_PS'] = [False, False, False]
    base_config_data['plot_H'] = [True, True, True]
    base_config_data['permute_H'] = [True, True, True]
    base_config_data['RH_start'] = [0, 0, 0]

    base_config_data['move_function'] = "sunfish_move"
    base_config_data['RP_true'] = [100, 280, 320, 479, 929, 60000]
    base_config_data['Rpst_true'] = [1, 1, 1, 1, 1, 1]
    base_config_data['RH_true'] = [0, 0, 0]

    base_config_data['permute_how_many'] = -1
    base_config_data['model'] = "sunfish_GRW"

    base_config_data['board_translation'] = "none"
    base_config_data['move_percentage_data'] = False

    config_data = union_dicts(base_config_data, model_config_data)

    match config_data[
        'model']:  # Load the model specified in the "base_config" file. Make sure the "model" field is set
        # correctly and that a model_result_string function is defined to properly store the results.
        case "sunfish_GRW":  # Sunfish Greedy Random Walk
            from irl_chess.models.sunfish_GRW import run_sunfish_GRW as model, \
                sunfish_native_result_string as model_result_string
        case "bayesian_optimisation":
            from irl_chess.models.bayesian_optimisation import run_bayesian_optimisation as model, \
                bayesian_model_result_string as model_result_string
        case "maia_pretrained":
            from irl_chess.models.maia_pretrained import run_maia_pre as model, \
                maia_pre_result_string as model_result_string
        case "BIRL":
            from irl_chess.models.BIRL import run_BIRL as model, \
                BIRL_result_string as model_result_string
        case _:
            raise Exception(f"No model found with the name {config_data['model']}")

    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)
    print(f'Output path: {out_path}')
    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')
    pgn_paths = ['data/raw/lichess_db_standard_rated_2019-01.pgn']

    # Loads in the chess format:
    chess_boards, player_moves = get_states(
        websites_filepath=websites_filepath,
        file_path_data=file_path_data,
        config_data=config_data,
        pgn_paths=pgn_paths,
        use_ply_range=False,
        out_path=out_path,
    )  # Boards in the sunfish format.

    val_df = load_maia_test_data(config_data['min_elo'], config_data['n_boards_val'])
    validation_set = list(zip([chess.Board(board_str) for board_str in val_df['board']], val_df['move']))
    
    for i in range(1, config_data['run_n_times']+1):
        out_path_i = join(out_path, f'run_{i}')
        os.makedirs(out_path_i, exist_ok=True)
        print(f'Output path: {out_path_i}')
        model(
            chess_boards=chess_boards,
            player_moves=player_moves,
            config_data=config_data,
            out_path=out_path_i,
            validation_set=validation_set
        )
