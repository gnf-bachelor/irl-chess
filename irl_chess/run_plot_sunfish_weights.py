import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from irl_chess.visualizations.visualize import plot_weights, sunfish_palette_name, maia_palette_name
from irl_chess.models.sunfish_GRW import val_sunfish_GRW


def run_loop(plot_weights, plot_accuracies, move_functions, initial_weights, elos, extensions=('', ), out_path=None, save_path=None):
    default_acc = [None for _ in elos]
    def_colors = sns.color_palette(maia_palette_name, len(elos))
    for move_function in move_functions:
        add_legend = True
        for initial_weight in initial_weights:
            for extension in extensions:
                color_palette = sns.color_palette(sunfish_palette_name, len(elos)*2)
                for i, elo in enumerate(elos):
                    base_config['min_elo'] = elo
                    base_config['max_elo'] = elo + 100
                    base_config['move_function'] = move_function
                    base_config['n_boards'] = 10000
                    base_config['plot_pst_char'] = []
                    base_config['plot_H'] = [False] * 3
                    base_config['alpha'] = .5
                    base_config['add_legend'] = add_legend
                    add_legend = False
                    base_config['plot_title'] = (f'Sunfish weights for multiple runs using {move_function.replace("_", " ")}s\n'
                                                 f'Using ELO bins {elos[0]}' + (f' and {elos[-1]} ' if len(elos)>1 else ''))
                    base_config["RP_start"][:5] = [initial_weight] * 5
                    base_config["model"] = 'sunfish_GRW'
                    base_config['permute_char'] = ["N", "B", "R", "Q"]

                    out_path = (create_result_path(base_config, model_config, sunfish_native_result_string) + extension) if out_path is None else out_path
                    if plot_weights:
                        save_path = join(os.getcwd(), 'results', 'plots', 'data_section', f'trace_plot_{move_function}_{initial_weights[-1]}_{elos[0]}_{elos[-1]}.svg') if save_path is None else save_path
                        try:
                            plot_R_weights(config_data=base_config,
                                   out_path=out_path,
                                   epoch=300,
                                   filename_addition='mod',# f'-{elo}-{elo+100}-{move_function}-gaussian',
                                   close_when_done=False,
                                   show=False,
                                   save_path=save_path)
                            print(move_function, initial_weight, extension, elo)
                            print(f'Were used!!')
                        except IndexError:
                            print(f'No weights saved at {out_path}')
                            print(move_function, initial_weight, extension, elo)

                    if plot_accuracies:
                        save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
                                         f'trace_plot_accuracies_{move_function}_{initial_weights[-1]}_{elos[0]}-{elos[-1]}.svg')

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
                        try:
                            df_best = pd.read_csv(join(out_path, 'weights', 'best_accuracies.csv', ))
                            df_temp = pd.read_csv(join(out_path, 'weights', 'temp_accuracies.csv', ))
                            best_acc_list = df_best.values.flatten()
                            temp_acc_list = df_temp.values.flatten()

                            if default_acc[i] is None and 'player' in move_function:
                                with open(join(out_path, 'boards_and_moves.pkl'), 'rb') as file:
                                    boards, moves = pickle.load(file)
                                validation_set = list(zip(boards, moves))

                                out_path_def = join(os.path.dirname(out_path), 'default_sunfish')
                                default_weight_path = join(out_path_def, 'weights')
                                os.makedirs(default_weight_path, exist_ok=True)
                                R_default = np.array(base_config['RP_true'])
                                R_pst = np.array(base_config['Rpst_true'])
                                df = pd.DataFrame(np.concatenate((R_default.reshape((-1,1)), R_pst.reshape((-1,1))), axis=1), columns=['Result', 'RpstResult'])
                                df.to_csv(join(default_weight_path, '0.csv'))

                                default_acc[i], _ = val_sunfish_GRW(validation_set=validation_set,
                                                                 out_path=os.path.dirname(default_weight_path),
                                                                 config_data=union_dicts(model_config, base_config),
                                                                 epoch=0,
                                                                 name=f'default-{elo}',
                                                                 use_player_moves=True)
                            elif default_acc[i] is None:
                                default_acc[i] = 1
                            print(default_acc)
                            plt.hlines(default_acc[i], xmin=0, xmax=len(best_acc_list), label=f'Default {elo}', linestyles='dashed', color=def_colors[i], alpha=0.7)

                            plt.title(f'Train Accuracy using {move_function.replace("_", " ")}s')
                            plt.plot(best_acc_list, label=f'Best {elo}', alpha=base_config['alpha']+.2, color=color_palette[2*i])
                            plt.plot(temp_acc_list, label=f'Epoch {elo}', alpha=base_config['alpha'], color=color_palette[2*i+1])
                            plt.ylabel('Accuracy')
                            plt.xlabel('Epoch')
                            plt.grid(True)
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                            plt.tight_layout()
                            print(f'Saved to {save_path}')
                            plt.savefig(save_path)
                        except FileNotFoundError:
                            print(f'No accuracies for {out_path}')
                            pass

        plt.show()
        plt.close()


if __name__ == '__main__':
    from irl_chess import load_config, create_result_path, sunfish_native_result_string, plot_R_weights, load_weights, \
    save_array, union_dicts

    base_config, model_config = load_config('sunfish_GRW')
    path_BIRL = r'C:\Users\fs\Downloads\BIRL'

    elos = (1100, )#1900)
    initial_weights = (101, 102)
    move_functions = ('sunfish_move', 'player_move', )
    extensions = ('', '_1')

    #save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
     #    f'trace_plot_{move_function}_{initial_weights[-1]}_{elos[0]}_{elos[-1]}.svg') if save_path is None else save_path

    #run_loop(plot_weights=True, plot_accuracies=False, move_functions=move_functions, initial_weights=initial_weights,
    #         elos=elos, extensions=extensions)

    run_loop(plot_weights=True, plot_accuracies=False, move_functions=move_functions, initial_weights=initial_weights, elos=elos, extensions=extensions)


