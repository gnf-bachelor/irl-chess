import os
from os.path import join

import pandas as pd

if __name__ == '__main__':
    from irl_chess import load_config, create_result_path, sunfish_native_result_string, plot_R_weights
    base_config, model_config = load_config('sunfish_GRW')

    elos = (1100, 1900)
    move_functions = ('sunfish_move', 'player_move')
    for elo in elos:
        for move_function in move_functions:
            base_config['min_elo'] = elo
            base_config['max_elo'] = elo + 100
            base_config['move_function'] = move_function
            base_config['n_boards'] = 10000

            out_path = create_result_path(base_config, model_config, sunfish_native_result_string)
            plot_R_weights(config_data=base_config,
                           out_path=out_path,
                           epoch=200,
                           filename_addition=f'-{elo}-{elo+100}-{move_function}-step')


