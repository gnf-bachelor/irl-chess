import os
from os.path import join
import chess.pgn
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos
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

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards, player_moves = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data) # Boards in the sunfish format.

    # testing if the data/ data gathering method is what is causing problems
    testing_2014 = False
    if testing_2014:
        def get_board_after_n_orig(game, n):
            board = game.board()
            for i, move in enumerate(game.mainline_moves()):
                board.push(move)
                if i == n:
                    break
            return board


        pgn = open("data/raw/lichess_db_standard_rated_2013-01.pgn")
        games = []
        n_games = 0
        n_boards_total = config_data['n_boards']
        n_boards_mid = n_boards_total//2
        n_boards_end = n_boards_total//2
        print('Getting games')
        while len(games) < n_boards_total:
            game = chess.pgn.read_game(pgn)
            if len(list(game.mainline_moves())) > 31:
                games.append(game)

        states_boards_mid = [get_board_after_n_orig(game, 15) for game in games[:n_boards_mid]]
        states_boards_end = [get_board_after_n_orig(game, 30) for game in games[:n_boards_end]]
        states_boards = states_boards_mid + states_boards_end
        sunfish_boards2 = [board2sunfish(board, eval_pos(board)) for board in states_boards]

    model(sunfish_boards=sunfish_boards, player_moves=player_moves, config_data=config_data, out_path=out_path)
