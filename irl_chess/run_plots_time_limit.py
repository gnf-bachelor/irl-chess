import os
import pickle
from collections import defaultdict
from os.path import join

import chess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from irl_chess.stat_tools.stat_tools import wilson_score_interval
from irl_chess.data.make_dataset import load_maia_test_data
from irl_chess.misc_utils.utils import union_dicts, create_default_sunfish
from irl_chess.models.sunfish_GRW import val_sunfish_GRW
from irl_chess.misc_utils.load_save_utils import save_array, get_states
from irl_chess.visualizations.visualize import sunfish_palette_name


def plot_time_limit(base_config, model_config, elos, time_limits):
    accuracies = [[] for el in time_limits]

    out_path = join(os.getcwd(), 'results', 'default_sunfish')
    print(f'Output path: {out_path}')
    create_default_sunfish(out_path=out_path, base_config=base_config)

    for j, time_limit in tqdm(enumerate(time_limits), desc='Time limits', total=len(time_limits)):
        for i, elo in tqdm(enumerate(elos), desc='ELOs', total=len(elos)):
            model_config['time_limit'] = time_limit
            base_config['min_elo'] = elo
            base_config['max_elo'] = elo + 100
            base_config['move_function'] = 'player_move'
            base_config['n_boards'] = 10000
            base_config['plot_pst_char'] = []
            base_config['plot_H'] = [False] * 3
            base_config['alpha'] = .5
            base_config['add_legend'] = True
            base_config['plot_title'] = (f'Sunfish default weights for different ELOs and time limits')

            val_df = load_maia_test_data(min_elo=elo, n_boards=base_config['n_boards'])
            boards, moves = [chess.Board(fen=fen) for fen in val_df['board']], val_df['move']
            validation_set = list(zip(boards, moves))

            acc, _ = val_sunfish_GRW(
                validation_set=validation_set,
                out_path=out_path,
                config_data=union_dicts(model_config, base_config),
                epoch=0,
                name=f'time_limit-{base_config["n_boards"]}-{time_limit}-{elo}',
                use_player_moves=True)
            accuracies[j].append(acc)

        save_filename = join(os.path.dirname(out_path), 'plots', 'time_limit', f'time_limits_{time_limits[0]}-{time_limit}-{elos[0]}-{elo}.svg')
        plot_accuracies_over_elo(accuracies, time_limits, elos, base_config['n_boards'], save_filename)


def plot_accuracies_over_elo(accuracies, time_limits, elos, n_boards, save_filename):
    plot_dict = defaultdict(lambda: [[], []])
    for time_limit, elo, accuracy in zip(time_limits, elos, accuracies):
        plot_dict[time_limit][0].append(accuracy)
        plot_dict[time_limit][1].append(elo)

    palette_sunfish = sns.color_palette(sunfish_palette_name, len(time_limits))

    plt.grid(axis='y', zorder=0)
    for i, (accs, color) in enumerate(zip(accuracies, palette_sunfish)):
        if accs:
            plt.plot(elos, accs, label=f'{time_limits[i]:.2}', color=color)
            lower, upper = wilson_score_interval(np.array(accs) * n_boards, n_boards)
            plt.fill_between(elos, lower, upper, color=color, alpha=0.15)

    plt.title('Model Accuracy by Player ELO and Time Limit')
    plt.xlabel('ELO')
    plt.ylabel('Accuracy')
    plt.legend(title='Seconds', loc='upper left', bbox_to_anchor=(1, 1))
    os.makedirs(os.path.dirname(save_filename), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_filename)
    print(f'Saved time limit plot to {save_filename}')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from irl_chess import load_config
    configs = load_config()
    time_limits = [0.01, 0.1, 0.2, 0.4, 0.75, 1., 1.5,]# 2.]
    time_limits.reverse()
    plot_time_limit(*configs, elos=range(1100, 2000, 100), time_limits=time_limits)
