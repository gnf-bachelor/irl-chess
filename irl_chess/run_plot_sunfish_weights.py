import json
import os
import pickle
from os.path import join

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from irl_chess.visualizations.visualize import plot_weights, sunfish_palette_name, maia_palette_name
from irl_chess.models.sunfish_GRW import val_sunfish_GRW
from irl_chess.misc_utils.utils import create_default_sunfish


def run_loop(plot_weights, plot_accuracies, move_functions, elos, extensions=('',), out_path=None, save_path=None,
             out_path_def_sunfish=None, add_legend=True, axes=None):
    default_acc = [None for _ in elos]
    def_colors = sns.color_palette(maia_palette_name, len(elos))
    for move_function in move_functions:
        for extension in extensions:
            color_palette = sns.color_palette(sunfish_palette_name, len(elos) * 2)
            for i, elo in enumerate(elos):
                base_config['min_elo'] = elo
                base_config['max_elo'] = elo + 100
                base_config['move_function'] = move_function
                base_config['n_boards'] = 10000
                base_config['alpha'] = .5
                base_config['add_legend'] = add_legend
                base_config["model"] = 'sunfish_GRW'
                base_config['permute_char'] = ["N", "B", "R", "Q"]

                out_path = (create_result_path(base_config, model_config,
                                               sunfish_native_result_string) + extension) if out_path is None else out_path
                if plot_weights:
                    save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
                                     f'trace_plot_{move_function}_{elos[0]}_{elos[-1]}.svg') if save_path is None else save_path
                    try:
                        plot_R_weights(config_data=base_config,
                                       out_path=out_path,
                                       epoch=300,
                                       filename_addition='mod',  # f'-{elo}-{elo+100}-{move_function}-gaussian',
                                       close_when_done=False,
                                       show=False,
                                       save_path=save_path)
                        print(move_function, extension, elo, f'Was used!!')
                        print(f'Saved to {save_path}')
                    except IndexError:
                        print(f'No weights saved at {out_path}')
                        print(move_function, extension, elo)

                if plot_accuracies:
                    save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
                                     f'trace_plot_accuracies_{move_function}__{elos[0]}-{elos[-1]}.svg') if save_path is None else save_path

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
                        df_temp = pd.read_csv(join(out_path, 'weights', 'temp_accuracies.csv', ))
                        temp_acc_list = df_temp.values.flatten()
                        print(f'Found temp accuracies')
                        try:
                            df_best = pd.read_csv(join(out_path, 'weights', 'best_accuracies.csv', ))
                            best_acc_list = df_best.values.flatten()
                            energies = None
                            print(f'Found best accuracies')
                        except FileNotFoundError:
                            print(f'No best accuracies found, looking for energy')
                            df_temp = pd.read_csv(join(out_path, 'weights', 'energies.csv'))
                            energies = df_temp.values.flatten()
                            best_acc_list = None
                            print(f'Found energies')
                        if default_acc[i] is None and 'player' in move_function:
                            with open(join(out_path, 'boards_and_moves.pkl'), 'rb') as file:
                                boards, moves = pickle.load(file)
                            validation_set = list(zip(boards, moves))


                            create_default_sunfish(out_path=out_path, base_config=base_config) if out_path_def_sunfish is None else None
                            default_acc[i], _ = val_sunfish_GRW(validation_set=validation_set,
                                                                out_path=out_path if out_path_def_sunfish is None else out_path_def_sunfish,
                                                                config_data=union_dicts(model_config, base_config),
                                                                epoch=0,
                                                                name=f'default-{elo}',
                                                                use_player_moves=True)
                        elif default_acc[i] is None:
                            default_acc[i] = 1

                        plt.title(f'Train Accuracy using {move_function.replace("_", " ")}s')
                        if best_acc_list is not None:
                            plt.plot(best_acc_list, label=f'Best {elo}', alpha=base_config['alpha'] + .2,
                                     color=color_palette[2 * i])
                        _, axs = plt.subplots(1, 1) if axes is None else (None, axes[0])

                        axs.hlines(default_acc[i], xmin=0, xmax=len(temp_acc_list), label=f'Default {elo}' if add_legend else None,
                                   linestyles='dashed', color=def_colors[i], alpha=0.7)
                        lns1 = axs.plot(temp_acc_list, label=f'Temp {elo}' if add_legend else None, alpha=base_config['alpha'],
                                 color=color_palette[2 * i + 1])

                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        axs.grid(True, axis='both')
                        if energies is not None:
                            ax2 = axs.twinx() if axes is None else axes[1]
                            lns2 = ax2.plot(energies, label=f'Energies {elo}' if add_legend else None, alpha=base_config['alpha'] + .2,
                                     color=color_palette[2 * i])
                            ax2.set_ylabel('Energy')
                            axes = axs, ax2
                            axs.legend(loc=(0, .75), ) if add_legend else None
                            ax2.legend(loc=(0, 0.88), ) if add_legend else None
                        else:
                            axes = axs, axs
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) if add_legend else None
                            axs.plot(best_acc_list, label=f'Best {elo}' if add_legend else None, alpha=base_config['alpha'],
                                 color=color_palette[2 * i + 1])

                        plt.tight_layout()
                        print(f'Saved to {save_path}')
                        plt.savefig(save_path)
                        add_legend = False
                    except FileNotFoundError:
                        print(f'No accuracies for {out_path}')
                        pass

    if axes is not None:
        return axes


def internal_loop(listdir, path_run, name, base_config):
    for do_weights in (True, False):
        for do_pst in (True, False):
            axes = None
            for i, filename in enumerate(listdir):
                add_legend = (len(listdir) - 2)
                if os.path.isdir(join(path_run, filename)) and not filename.endswith('plots'):
                    save_path = join(path_run, 'plots',
                                     f'trace_plot_{name}_{"pst" if do_pst else "pieces"}_{"weights" if do_weights else "accuracies"}.svg')
                    path_out = join(path_run, filename, )

                    base_config['plot_pst_char'] = ["P", "N", "B", "R", "Q", "K"] if do_pst else []
                    base_config['plot_H'] = [False] * 3
                    base_config['plot_char'] = ["P", "N", "B", "R", "Q", ] if not do_pst else []
                    base_config['plot_title'] = (
                        f'Sunfish {("pst" if do_pst else "piece") if do_weights else ""} '
                        f'{("weights" if do_weights else "accuracies and energies")} '
                        f'for multiple runs\n using Sunfish moves '
                        f'and {name}')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    axes = run_loop(plot_weights=do_weights,
                                    plot_accuracies=not do_weights,
                                    move_functions=('sunfish_move',),
                                    elos=(1900,),
                                    extensions=extensions,
                                    out_path=path_out,
                                    save_path=save_path,
                                    out_path_def_sunfish=path_def_sunfish,
                                    add_legend=i == add_legend,
                                    axes=axes)
            plt.show()
            plt.close()


if __name__ == '__main__':
    from irl_chess import load_config, create_result_path, sunfish_native_result_string, plot_R_weights, load_weights, \
        save_array, union_dicts
    # Run twice if plot folders are not created yet, very scuffed yes.

    options = ['gpw_8', 'SunfVal', 'alpha_beta', 'BIRLall', ]

    path_ahh = r'C:\Users\fs\Downloads\ahh'
    path_def_sunfish = join(path_ahh, 'default_sunfish')
    base_config, model_config = load_config('sunfish_GRW')

    elos = (1100, 1900)
    move_functions = ('player_move', 'sunfish_move',)
    extensions = ('',)

    for opt_index in [1]:
        count = 0
        if options[opt_index] == 'gpw_8':
            for elo in elos:
                base_config['plot_pst_char'] = []
                base_config['plot_H'] = [False] * 3
                base_config['plot_char'] = ["P", "N", "B", "R", "Q"]
                for do_weights in (False,):
                    for move_function in move_functions:
                        base_config['plot_title'] = (
                                f'Sunfish weights for multiple runs using {move_function.replace("_", " ")}s\n'
                                f'Using ELO bins {elos[0]}' + (f' and {elos[-1]} ' if len(elos) > 1 else ''))
                        path_run = join(path_ahh, move_function, f'{elo}', )
                        listdir = os.listdir(path_run)
                        for i, filename in enumerate(listdir):
                            add_legend = i == (len(listdir) - 2)
                            if os.path.isdir(join(path_run, filename)) and not filename.endswith('plots'):
                                save_path = join(path_run, 'plots',
                                                 f'trace_plot_pieces_{move_function}_{elo}_{"weights" if do_weights else "accuracies"}.svg')
                                path_out = join(path_run, filename)
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                run_loop(plot_weights=do_weights,
                                         plot_accuracies=not do_weights,
                                         move_functions=(move_function,),
                                         elos=(elo,),
                                         extensions=extensions,
                                         out_path=path_out,
                                         save_path=save_path,
                                         out_path_def_sunfish=path_def_sunfish,
                                         add_legend=add_legend)
                        count += 1
                        print(count)

                        plt.show()
                        plt.close()
        elif options[opt_index] == 'SunfVal':
            for do_weights in (True, False):
                path_run = join(path_ahh, 'SunfVal')
                for do_pst in (True, False):
                    for filename in os.listdir(path_run):
                        if os.path.isdir(join(path_run, filename)) and not filename.endswith('plots'):
                            for i in range(1, 3):
                                save_path = join(path_run, filename, 'plots',
                                                 f'trace_plot_SunfVal_{"pst" if do_pst else "pieces"}_{"weights" if do_weights else "accuracies"}.svg')
                                path_out = join(path_run, filename, f'run_{i}')
                                with open(join(path_run, filename, 'configs', 'base_config.json'), 'rb') as file:
                                    base_config = json.load(file)
                                base_config['plot_pst_char'] = ["P", "N", "B", "R", "Q", "K"] if do_pst else []
                                base_config['plot_H'] = [False] * 3
                                base_config['plot_char'] = ["P", "N", "B", "R", "Q", ] if not do_pst else []
                                base_config['plot_title'] = (
                                        f'Sunfish {("pst" if do_pst else "piece") if do_weights else ""} {("weights" if do_weights else "accuracies")} for multiple runs using Sunfish moves\n'
                                        f'Using ELO bins {elos[0]}' + (f' and {elos[-1]} ' if len(elos) > 1 else ''))
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                run_loop(plot_weights=do_weights,
                                         plot_accuracies=not do_weights,
                                         move_functions=('sunfish_move',),
                                         elos=(1900,),
                                         extensions=extensions,
                                         out_path=path_out,
                                         save_path=save_path,
                                         out_path_def_sunfish=path_def_sunfish)
                        plt.show()
                        plt.close()
        elif options[opt_index] == 'alpha_beta':
            path_run = join(path_ahh, options[opt_index])
            listdir = os.listdir(path_run)
            internal_loop(listdir, path_run, options[opt_index], base_config)
        elif options[opt_index] == 'BIRLall':
            for path_run in [join(path_ahh, options[opt_index]), join(path_ahh, options[opt_index], '2')]:
                listdir = os.listdir(path_run)
                internal_loop(listdir, path_run, options[opt_index][:-3], base_config)


