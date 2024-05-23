import os
from collections import defaultdict
from os.path import join

import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import seaborn as sns
from irl_chess.visualizations import sunfish_palette_name


def plot_accuracies_over_elo(accuracies, player_elos, model_elos, n_boards, model_names):
    plot_dict = defaultdict(lambda: [[], []])
    for model_elo, model_name, accuracy, elo in zip(model_elos, model_names, accuracies, player_elos):
        name = f'{model_name[0].upper() + model_name[1:]} ELO {model_elo}' if model_elo else model_name
        plot_dict[name][0].append(accuracy)
        plot_dict[name][1].append(elo)

    n_maia_colors = len([name for name in plot_dict.keys() if 'maia' in name.lower()])
    n_sunfish_colors = len([name for name in plot_dict.keys() if 'maia' not in name.lower()])
    palette_maia = sns.color_palette("flare", n_maia_colors)
    palette_sunfish = sns.color_palette(sunfish_palette_name, n_sunfish_colors)

    plt.grid(axis='y', zorder=0)
    idxs = [0, 0]
    for idx, (name, (accs, elos_)) in enumerate(plot_dict.items()):
        is_maia = 'maia' in name.lower()
        color = palette_maia[idxs[is_maia]] if is_maia else palette_sunfish[idxs[is_maia]]
        plt.plot(elos_, accs, label=name, color=color)
        lower, upper = wilson_score_interval(np.array(accs) * n_boards, n_boards)
        plt.fill_between(elos_, lower, upper, color=color, alpha=0.1)
        idxs[is_maia] += 1

    plt.title('Model Accuracy by Player ELO')
    plt.xlabel('Player ELO')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    os.makedirs(f'results/plots/models_by_elo', exist_ok=True)
    plt.tight_layout()
    plt.savefig(f'results/plots/models_by_elo/models_by_elo_{model_elos[0]}_{model_elos[-1]}_{n_boards}.svg')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from irl_chess import load_maia_network, make_maia_test_csv, load_maia_test_data, val_sunfish_GRW, load_config, \
    create_result_path, sunfish_native_result_string, union_dicts, wilson_score_interval

    elos_players, accuracies, model_names_list, model_elos = [], [], [], []

    n_boards = 5000
    model_names = ['sunfish', 'maia', ]
    maia_range = (1100, 2000)  # incl. excl.
    sunfish_elo_epoch = {1100: 100, 1900: 100, 'Default Sunfish': 0, }
    player_range = (1100, 2000)  # incl. excl.
    # make_maia_test_csv(destination, min_elo=player_range[0], max_elo=player_range[1], n_boards=n_boards)
    config_data_base, config_data_sunfish = load_config('sunfish_GRW')

    range_maia = [el for el in range(maia_range[0], maia_range[1], 100)]
    range_sunfish = [el for el in sunfish_elo_epoch.keys()]
    player_elos_iter = [el for el in range(player_range[0], player_range[1], 100)]

    for model_name in tqdm(model_names, desc='Models'):
        model_range = range_maia if model_name == 'maia' else range_sunfish
        n_total = len(model_range) * len(player_range)
        count = 0
        csv_path = (f'results/models_over_elo/{model_name}_{model_range[0]}_{model_range[-1]}_{n_total}_'
                    f'{player_elos_iter[0]}_{player_elos_iter[-1]}_{n_boards}.csv')
        if os.path.exists(csv_path):
            print(f'Loading {csv_path} and continuing from there')
            df = pd.read_csv(csv_path)
            accuracies += list(df[f'Accuracy'])
            elos_players += list(df['ELO Players'])
            model_names_list += list(df['Model name'])
            model_elos += list(df['ELO model'])
        else:
            print(f'Path {csv_path} does not exist, will run analysis from here')
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        for elo_model in tqdm(model_range, desc=f'ELO {model_name}'):
            model = load_maia_network(elo=elo_model, parent='irl_chess/maia_chess/') if model_name == 'maia' else None
            for elo_player in tqdm(player_elos_iter, desc='ELO Players'):
                count += 1
                if count <= len(accuracies):
                    continue
                val_df = load_maia_test_data(elo_player, n_boards=n_boards)
                validation_set = list(zip([chess.Board(fen=fen) for fen in val_df['board']], val_df['move']))
                if model_name == 'maia':
                    acc_sum = 0
                    # page 4 of the paper states that moves with less than 30 seconds left have been discarded
                    for board, move_true in tqdm(validation_set, desc='Maia Moves', total=n_boards):
                        move_maia = model.getTopMovesCP(board, 1)[0][0]
                        acc_sum += move_maia == move_true
                    acc_model = (acc_sum / val_df.shape[0])
                elif model_name == 'sunfish':
                    config_data_base['model'] = "sunfish_GRW"
                    config_data_base['move_function'] = "player_move"
                    out_path = create_result_path(config_data_base,
                                                  model_config_data=config_data_sunfish,
                                                  model_result_string=sunfish_native_result_string)
                    if elo_model == 'Default Sunfish':
                        out_path = join(os.path.dirname(out_path), 'default_sunfish')
                        default_weight_path = join(out_path, 'weights')
                        os.makedirs(default_weight_path, exist_ok=True)
                        R_default = np.array(config_data_base['RP_true'])
                        R_pst = np.array(config_data_base['Rpst_true'])
                        df = pd.DataFrame(np.concatenate((R_default.reshape((-1,1)), R_pst.reshape((-1,1))), axis=1), columns=['Result', 'RpstResult'])
                        df.to_csv(join(default_weight_path, '0.csv'))
                    else:
                        config_data_base['min_elo'] = elo_model
                        config_data_base['max_elo'] = elo_model + 100
                    acc_model, _ = val_sunfish_GRW(
                        validation_set,
                        use_player_moves=True,
                        config_data=union_dicts(config_data_base, config_data_sunfish),
                        epoch=sunfish_elo_epoch[elo_model],
                        out_path=out_path,
                        name=n_boards
                    )
                else:
                    print('Invalid model, will crash')

                model_names_list.append(model_name)
                model_elos.append(elo_model)
                accuracies.append(acc_model)
                elos_players.append(elo_player)
                print(f'ELO {model_name}: {elo_model}, ELO Players: {elo_player}, Accuracy: {acc_model}')
                plot_accuracies_over_elo(accuracies=accuracies,
                                         player_elos=elos_players,
                                         n_boards=n_boards,
                                         model_names=model_names_list,
                                         model_elos=model_elos)

                df_out = pd.DataFrame(np.array([
                    np.array(model_names_list).flatten(),
                    np.array(model_elos).flatten(),
                    np.array(elos_players).flatten(),
                    np.array(accuracies).flatten(),
                ]).T,
                                      columns=['Model name', f'ELO model', 'ELO Players', f'Accuracy',])
                df_out.to_csv(csv_path, index=False)
            model = None
    plot_accuracies_over_elo(accuracies=accuracies,
                             player_elos=elos_players,
                             n_boards=n_boards,
                             model_names=model_names_list,
                             model_elos=model_elos)