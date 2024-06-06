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
             out_path_def_sunfish=None, axes=None, plot_def_sunfish=True):
    default_acc = [None for _ in elos]
    def_colors = sns.color_palette(maia_palette_name, len(elos))
    for move_function in move_functions:
        for extension in extensions:
            color_palette = sns.color_palette(sunfish_palette_name, len(elos) * 2)
            for i, elo in enumerate(elos):
                add_legend = not plt.fignum_exists(1)
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
                        print(f'Saved to {save_path}')
                    except IndexError:
                        print(f'No weights saved at {out_path}')
                        print(move_function, extension, elo)

                if plot_accuracies:
                    save_path = join(os.getcwd(), 'results', 'plots', 'data_section',
                                     f'trace_plot_accuracies_{move_function}_{elos[0]}-{elos[-1]}.svg') if save_path is None else save_path

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
                            print(f'No best accuracies found, looking for a_energy')
                            try:
                                df_temp = pd.read_csv(join(out_path, 'weights', 'a_energies.csv'))
                                print(f'Found a_energies')
                            except FileNotFoundError:
                                print(f'No a_energies trying just energies')
                                df_temp = pd.read_csv(join(out_path, 'weights', 'energies.csv'))
                            energies = df_temp.values.flatten()
                            best_acc_list = None
                            print(f'Found energies')
                        if default_acc[i] is None and 'player' in move_function:
                            with open(join(out_path, 'boards_and_moves.pkl'), 'rb') as file:
                                boards, moves = pickle.load(file)
                            validation_set = list(zip(boards, moves))

                            create_default_sunfish(out_path=out_path,
                                                   base_config=base_config) if out_path_def_sunfish is None else None
                            default_acc[i], _ = val_sunfish_GRW(validation_set=validation_set,
                                                                out_path=out_path if out_path_def_sunfish is None else out_path_def_sunfish,
                                                                config_data=union_dicts(model_config, base_config),
                                                                epoch=0,
                                                                name=f'default-{elo}',
                                                                use_player_moves=True)
                        elif default_acc[i] is None:
                            default_acc[i] = 1

                        plt.title(
                            base_config.get('plot_title', f'Train Accuracy using {move_function.replace("_", " ")}s'))
                        if best_acc_list is not None:
                            plt.plot(best_acc_list, label=f'Best {elo}', alpha=base_config['alpha'] + .2,
                                     color=color_palette[2 * i])
                        _, axs = plt.subplots(1, 1) if axes is None else (None, axes[0])

                        axs.hlines(default_acc[i], xmin=0, xmax=len(temp_acc_list),
                                   label=f'Default {elo}' if add_legend else None,
                                   linestyles='dashed', color=def_colors[i], alpha=0.7) if plot_def_sunfish else None
                        lns1 = axs.plot(temp_acc_list, label=f'Temp {elo}' if add_legend else None,
                                        alpha=base_config['alpha'],
                                        color=color_palette[2 * i + 1])

                        plt.ylabel('Accuracy')
                        plt.xlabel('Epoch')
                        axs.grid(True, axis='both')
                        if energies is not None:
                            ax2 = axs.twinx() if axes is None else axes[1]
                            lns2 = ax2.plot(energies, label=f'Energies {elo}' if add_legend else None,
                                            alpha=base_config['alpha'] + .2,
                                            color=color_palette[2 * i])
                            ax2.set_ylabel('Energy')
                            axes = axs, ax2
                            axs.legend(loc=(0, .75), ) if add_legend else None
                            ax2.legend(loc=(0, 0.88), ) if add_legend else None
                        else:
                            axes = axs, axs
                            plt.legend(loc='upper left', bbox_to_anchor=(1, 1)) if add_legend else None
                            axs.plot(best_acc_list, label=f'Best {elo}' if add_legend else None,
                                     alpha=base_config['alpha'],
                                     color=color_palette[2 * i + 1])

                        plt.tight_layout()
                        print(f'Saved to {save_path}\n Not adding legend to next plot')
                        plt.savefig(save_path)
                    except FileNotFoundError:
                        print(f'No accuracies for {out_path}')
                        pass

    if axes is not None:
        return axes


def internal_loop(listdir, path_run, name_title, base_config,
                  name_figure=None,
                  second_savepath=None,
                  plot_def_sunfish=True,
                  move_functions=('sunfish_move',),
                  invalid_folder_names=('plots', 'configs', 'default_sunfish'),
                  do_weights_options=(True, False),
                  do_pst_options=(True, False),
                  elos=(1900,)):
    if name_figure is None: name_figure = name_title
    for do_weights in do_weights_options:
        for do_pst in do_pst_options:
            axes = None
            count = 0
            for i, filename in enumerate(listdir):
                if os.path.isdir(join(path_run, filename)) and not any(
                        [filename.endswith(el) for el in invalid_folder_names]):
                    plot_name = (f'trace_plot_{name_figure}_{("pst_" if do_pst else "piece_") if do_weights else ""}'
                                 f'{("weights" if do_weights else "accuracies")}.svg')
                    save_path = join(path_run, 'plots', plot_name)
                    path_out = join(path_run, filename, )

                    base_config['plot_pst_char'] = ["P", "N", "B", "R", "Q", "K"] if do_pst else []
                    base_config['plot_H'] = [False] * 3
                    base_config['plot_char'] = ["P", "N", "B", "R", "Q", ] if not do_pst else []
                    base_config['plot_title'] = base_config.get('plot_title',
                                                                f'Sunfish {("pst" if do_pst else "piece") if do_weights else ""} '
                                                                f'{("weights" if do_weights else "accuracies")} '
                                                                f'for multiple runs\n using '
                                                                f'the {name_title} Algorithm')
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    axes = run_loop(plot_weights=do_weights,
                                    plot_accuracies=not do_weights,
                                    move_functions=move_functions,
                                    elos=elos,
                                    extensions=extensions,
                                    out_path=path_out,
                                    save_path=save_path,
                                    out_path_def_sunfish=path_def_sunfish,
                                    axes=axes, )
                    count += 1
            if second_savepath is not None:
                plt.savefig(join(second_savepath, plot_name))
            print(f'Showing now after adding {count} runs to plot')
            plt.show()
            plt.close()


if __name__ == '__main__':
    from irl_chess import load_config, create_result_path, sunfish_native_result_string, plot_R_weights, load_weights, \
        save_array, union_dicts

    # Run twice if plot folders are not created yet, very scuffed yes.

    options = {
        0: 'gpw_8',
        1: 'gpw_pst',
        2: 'SunfVal',
        3: 'alpha_beta',
        4: 'BIRLall',
        5: 'BIRLPlayerAppendix',
        6: 'BirlPlaceholder',
        7: 'BirlReal'
    }

    path_ahh = r'C:\Users\fs\Downloads\ahh'
    path_def_sunfish = join(path_ahh, 'default_sunfish')
    base_config, model_config = load_config('sunfish_GRW')

    move_functions = ('player_move', 'sunfish_move',)
    extensions = ('',)

    for opt_index in range(1, 2):
        count = 0
        if options[opt_index] == 'gpw_8':
            elos = (1100, 1900)
            for elo in elos:
                add_legend = True
                for move_function in move_functions:
                    base_config['plot_title'] = (
                        f'Sunfish weights for multiple runs using {move_function.replace("_", " ")}s\n'
                        f'Using ELO bin {elo}')
                    path_run = join(path_ahh, move_function, f'{elo}', )
                    listdir = os.listdir(path_run)
                    second_savepath = join(path_ahh, 'GPW8_plots')
                    os.makedirs(second_savepath, exist_ok=True)
                    internal_loop(listdir=listdir,
                                  path_run=path_run,
                                  move_functions=(move_function,),
                                  name_title='GPW',
                                  name_figure=f'GPW_{elo}_{move_function}',
                                  second_savepath=second_savepath,
                                  base_config=base_config, do_pst_options=(False,), elos=(elo,))
                    add_legend = False
        if options[opt_index] == 'gpw_pst':
            elos = (1100, 1900)
            for elo in elos:
                for move_function in move_functions:
                    base_config['plot_title'] = (
                        f'Sunfish weights for multiple runs using {move_function.replace("_", " ")}s\n'
                        f'Using ELO {elo}')
                    path_run = join(path_ahh, 'pst', move_function, f'{elo}', )
                    listdir = os.listdir(path_run)
                    second_savepath = join(path_ahh, 'GPW_pst_plots')
                    os.makedirs(second_savepath, exist_ok=True)
                    internal_loop(listdir=listdir,
                                  path_run=path_run,
                                  move_functions=(move_function,),
                                  name_title='GPW',
                                  name_figure=f'GPW_{elo}_{move_function}',
                                  second_savepath=second_savepath,
                                  base_config=base_config, do_pst_options=(True,), elos=(elo,), )

        elif options[opt_index] == 'SunfVal':
            path_run_base = join(path_ahh, 'SunfVal')
            for filename in os.listdir(path_run_base):
                if not filename.endswith('plots'):
                    path_run = join(path_run_base, filename)
                    listdir = os.listdir(path_run)
                    second_savepath = join(path_ahh, 'SunfValPlots')
                    os.makedirs(second_savepath, exist_ok=True)

                    internal_loop(listdir=listdir, path_run=path_run, name_title='Sunfish', base_config=base_config, second_savepath=second_savepath)
            print(f'Done internal')
            internal_loop(listdir, path_run, options[opt_index], base_config)
        elif options[opt_index] == 'BIRLall':
            for path_run in [join(path_ahh, options[opt_index]), join(path_ahh, options[opt_index], '2')]:
                listdir = os.listdir(path_run)
                second_savepath = join(path_ahh, 'BIRLall_plots')
                os.makedirs(second_savepath, exist_ok=True)
                internal_loop(listdir, path_run, options[opt_index][:-3], base_config, second_savepath=second_savepath)
        elif options[opt_index] in ['BIRLPlayerAppendix', ]:
            path_run = join(path_ahh, options[opt_index])
            listdir = os.listdir(path_run)
            internal_loop(listdir, path_run, 'BIRL', base_config, move_functions=('player_move',),
                          name_figure=f'BIRL_player', plot_def_sunfish=False)
        elif options[opt_index] in ['BirlPlaceholder', ]:
            path_run = join(path_ahh, options[opt_index])
            listdir = os.listdir(path_run)
            internal_loop(listdir, path_run, 'BIRL', base_config, plot_def_sunfish=False,
                          name_figure=options[opt_index])
        elif options[opt_index] in ['BirlReal', ]:
            path_run = join(path_ahh, options[opt_index])
            listdir = os.listdir(path_run)
            second_savepath = join(path_ahh, 'BIRLRealPlots')
            os.makedirs(second_savepath, exist_ok=True)

            internal_loop(listdir, path_run, 'BIRL', base_config, plot_def_sunfish=False,
                          name_figure=options[opt_index], second_savepath=second_savepath)
        elif options[opt_index] in ['alpha_beta', ]:
            path_run = join(path_ahh, options[opt_index])
            listdir = os.listdir(path_run)
            second_savepath = join(path_ahh, 'alpha-beta-plots')
            os.makedirs(second_savepath, exist_ok=True)
            internal_loop(listdir, path_run, options[opt_index], base_config, second_savepath=second_savepath)
