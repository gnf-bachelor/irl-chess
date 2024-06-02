import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from irl_chess.visualizations.visualize import plot_weights, sunfish_palette_name

def run_loop(plot_weights, plot_accuracies, move_functions, initial_weights, elos):
    for move_function in move_functions:
        for initial_weight in initial_weights:
            color_palette = sns.color_palette(sunfish_palette_name, len(elos)*2)
            for i, elo in enumerate(elos):
                base_config['min_elo'] = elo
                base_config['max_elo'] = elo + 100
                base_config['move_function'] = move_function
                base_config['n_boards'] = 10000
                base_config['plot_pst_char'] = []
                base_config['plot_H'] = [False] * 3
                base_config['alpha'] = .5
                base_config['add_legend'] = (elo == elos[-1]) and (move_function == move_functions[-1])
                base_config['plot_title'] = (f'Sunfish weights for multiple runs using {move_function.replace("_", " ")}s\n'
                                             f'Using ELO bins {elos[0]} and {elos[-1]} ')
                base_config["RP_start"][:5] = [initial_weight] * 5

                out_path = create_result_path(base_config, model_config, sunfish_native_result_string)
                if plot_weights:
                    plot_R_weights(config_data=base_config,
                               out_path=out_path,
                               epoch=500,
                               filename_addition='mod',# f'-{elo}-{elo+100}-{move_function}-gaussian',
                               close_when_done=False,
                               show=False,
                               save_path=join(os.getcwd(), 'results', 'plots', 'data_section', f'trace_plot_{move_function}_{base_config["RP_start"][1]}.svg'))
                if plot_accuracies:
                    save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
                                     f'trace_plot_accuracies_{move_function}_{base_config["RP_start"][1]}.svg')

                    try:
                        acc_path = join(out_path, 'accuracies')
                        with open(acc_path, 'rb') as f:
                            acc = pickle.load(f)
                        best_acc_list = [el[0] for el in acc]
                        temp_acc_list = [el[1] for el in acc]
                        save_array(best_acc_list, "best_accuracies", out_path)
                        save_array(temp_acc_list, "temp_accuracies", out_path)
                    except FileNotFoundError:
                        print(f'No pickled accuracies, using csvs')
                        pass
                    df_best = pd.read_csv(join(out_path, 'weights', 'best_accuracies.csv', ))
                    df_temp = pd.read_csv(join(out_path, 'weights', 'temp_accuracies.csv', ))
                    best_acc_list = df_best.values.flatten()
                    temp_acc_list = df_temp.values.flatten()

                    plt.title(f'Train Accuracy using {move_function.replace("_", " ")}s')
                    plt.plot(best_acc_list, label=f'Best {elo}', alpha=base_config['alpha']+.2, color=color_palette[2*i])
                    plt.plot(temp_acc_list, label=f'Epoch {elo}', alpha=base_config['alpha'], color=color_palette[2*i+1])
                    plt.ylabel('Accuracy')
                    plt.xlabel('Epoch')
                    plt.grid(True)
                    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                    plt.tight_layout()
                    plt.savefig(save_path)

        plt.show()
        plt.close()


if __name__ == '__main__':
    from irl_chess import load_config, create_result_path, sunfish_native_result_string, plot_R_weights, load_weights, \
    save_array

    base_config, model_config = load_config('sunfish_GRW')

    elos = (1100, 1900)
    initial_weights = (101, )
    move_functions = ('sunfish_move', 'player_move', )
    for move_function in move_functions:
        run_loop(plot_weights=False, plot_accuracies=True, move_functions=(move_function, ), initial_weights=initial_weights, elos=elos)


