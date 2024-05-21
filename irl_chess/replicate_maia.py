import os
import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time


def plot_accuracies(accuracies_maia, accuracies_sunfish, elos, maia_elos, n_moves):
    for elos_, accs_maia, accs_sunfish, maia_elo in zip(elos, accuracies_maia, accuracies_sunfish, maia_elos):
        plt.plot(elos_, accs_maia, label=f'Maia ELO {maia_elo}')
        if np.nan not in accs_sunfish:
            plt.plot(elos_, accs_sunfish, label=f'Sunfish ELO {maia_elo}')

    plt.title('Maia Accuracy by Player ELO')
    plt.xlabel('Player ELOs')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    os.makedirs('results/plots/maia_replication_accuracy', exist_ok=True)
    plt.savefig(f'results/plots/maia_replication_accuracy/{maia_elos[0]}_{maia_elos[-1]}_{n_moves}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from irl_chess import load_maia_network, make_maia_test_csv, load_maia_test_data, val_sunfish_GRW, load_config, \
    create_result_path, sunfish_native_result_string, union_dicts

    # This must be run from the main irl-chess folder
    t = time()
    destination = 'data/raw/maia-chess-testing-set.csv.bz2'

    print(f'Took {time() - t:.2f} seconds to download maia dataset')

    elos_players, accuracies_maia, accuracies_sunfish, maia_elos = [], [], [], []
    n_boards = 5000
    maia_range = (1300, 2000)  # incl. excl.
    sunfish_elo_epoch = {1100: 100, 1300: 90, 1500: 90, 1700: 90, 1900: 100}
    player_range = (1000, 1900)  # incl. excl.
    # make_maia_test_csv(destination, min_elo=player_range[0], max_elo=player_range[1], n_boards=n_boards)
    config_data_base, config_data_sunfish = load_config()

    for elo_maia in tqdm(range(maia_range[0], maia_range[1], 100), desc='ELO Maia'):
        accuracies_maia.append([])
        accuracies_sunfish.append([])
        elos_players.append([])
        maia_elos.append(elo_maia)
        csv_path = f'results/maia_replication/{maia_elos[0]}_{maia_elos[-1]}_{n_boards}.csv'
        while not accuracies_sunfish[-1]:
            print(f'Sunfish elo: {elo_maia} starting now')
            for elo_player in tqdm(range(player_range[0], player_range[1], 100), desc='ELO Players'):

                model = load_maia_network(elo=elo_maia, parent='irl_chess/maia_chess/')
                val_df = load_maia_test_data(elo_player, n_boards=n_boards)

                acc_sum = 0
                # page 4 of the paper states that moves with less than 30 seconds left have been discarded
                validation_set = list(zip([chess.Board(fen=fen) for fen in val_df['board']], val_df['move']))
                for board, move_true in tqdm(validation_set, desc='Maia Moves', total=n_boards):
                    move_maia = model.getTopMovesCP(board, 1)[0][0]
                    acc_sum += move_maia == move_true
                accuracies_maia[-1].append(acc_sum / val_df.shape[0])
                elos_players[-1].append(elo_player)

                if elo_maia in sunfish_elo_epoch.keys():
                    config_data_base['min_elo'] = elo_maia
                    config_data_base['max_elo'] = elo_maia + 100
                    out_path = create_result_path(config_data_base,
                                                  model_config_data=config_data_sunfish,
                                                  model_result_string=sunfish_native_result_string)
                    acc_sunfish, _ = val_sunfish_GRW(
                        validation_set,
                        use_player_moves=True,
                        config_data=union_dicts(config_data_base, config_data_sunfish),
                        epoch=sunfish_elo_epoch[elo_maia],
                        out_path=out_path
                    )
                    accuracies_sunfish[-1].append(acc_sunfish)
                else:
                    accuracies_sunfish[-1].append(np.nan)

                print(f'Elo Maia: {elo_maia}, Elo Players: {elo_player}, Accuracy: {acc_sum / val_df.shape[0]}')
            plot_accuracies(accuracies_maia, accuracies_sunfish, elos_players, maia_elos, n_moves=n_boards)
            df_out = pd.DataFrame(np.array([
                np.array([[maia_elo] * len(elos_players[-1]) for maia_elo in maia_elos]).flatten(),
                np.array(elos_players).flatten(),
                np.array(accuracies_maia).flatten(),
                np.array(accuracies_sunfish).flatten()
            ]).T,
                                  columns=['Elo Maia', 'Elo Players', 'Accuracy Maia', 'Accuracy Sunfish'])
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            df_out.to_csv(csv_path)
