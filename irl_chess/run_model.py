import os
from os.path import join
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, create_result_path, get_states

if __name__ == '__main__':
    from irl_chess import load_maia_test_data
    fix_cwd()
    base_config_data, model_config_data = load_config()
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

        case _:
            raise Exception(f"No model found with the name {config_data['model']}")

    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')
    pgn_paths = ['data/raw/lichess_db_standard_rated_2017-11.pgn']

    # Loads in the chess format:
    chess_boards, player_moves = get_states(
        websites_filepath=websites_filepath,
        file_path_data=file_path_data,
        config_data=config_data,
        pgn_paths=pgn_paths,
        use_ply_range=False,
    )  # Boards in the sunfish format.

    val_df = load_maia_test_data(config_data['min_elo'], config_data['n_boards_val'])
    validation_set = list(zip(val_df['move'], val_df['board']))

    model(
        chess_boards=chess_boards,
        player_moves=player_moves,
        config_data=config_data,
        out_path=out_path,
        validation_set=validation_set
    )
