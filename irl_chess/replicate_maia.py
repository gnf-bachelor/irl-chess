import os
import chess
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time


def plot_accuracies(accuracies, elos, maia_elos, n_moves):
    for elos_, accs, maia_elo in zip(elos, accuracies, maia_elos):
        plt.plot(elos_, accs, label=f'Maia ELO {maia_elo}')

    plt.title('Maia Accuracy by Player ELO')
    plt.xlabel('Player ELOs')
    plt.ylabel('Accuracy')
    plt.legend()
    os.makedirs('results/plots', exist_ok=True)
    plt.savefig(f'results/plots/maia_replication_accuracy_{maia_elos[0]}_{maia_elos[-1]}_{n_moves}.png')
    plt.show()
    plt.close()


if __name__ == '__main__':
    from irl_chess import download_file 
    from irl_chess.maia_chess import load_maia_network
    # This must be run from the main irl-chess folder
    t = time()
    destination = 'data/maia-chess-testing-set.csv.bz2'
    url = r'https://csslab.cs.toronto.edu/data/chess/kdd/maia-chess-testing-set.csv.bz2'
    if not os.path.exists(destination):
        download_file(url=url, destination=destination)

    df = pd.read_csv(destination)
    print(f'Took {time() - t:.2f} seconds to download and load maia dataset')
    elos_players, accuracies, maia_elos = [], [], []
    n_boards = 5000
    maia_range = (1100, 2000)   # incl. excl.
    player_range = (1000, 1900) # incl. excl.

    for elo_maia in tqdm(range(maia_range[0], maia_range[1], 100), desc='ELO Maia'):
        accuracies.append([])
        elos_players.append([])
        maia_elos.append(elo_maia)
        for elo_player in tqdm(range(player_range[0], player_range[1], 100), desc='ELO Players'):
            model = load_maia_network(elo=elo_maia, parent='irl_chess/maia_chess/')
            val_df = df[(elo_player < df['opponent_elo']) & (df['white_elo'] < (elo_player + 100))]
            n_boards = val_df.shape[0] if n_boards is None else n_boards
            val_df = val_df[(10 <= val_df['move_ply'])][:n_boards]
            acc_sum = 0
            # page 4 of the paper states that moves with less than 30 seconds left have been discarded
            for move_true, fen in tqdm(zip(val_df['move'], val_df['board']), desc='Moves', total=n_boards):
                board = chess.Board(fen=fen)
                move_maia = model.getTopMovesCP(board, 1)[0][0]
                acc_sum += move_maia == move_true
            accuracies[-1].append(acc_sum / val_df.shape[0])
            elos_players[-1].append(elo_player)
            print(f'Elo Maia: {elo_maia}, Elo Players: {elo_player}, Accuracy: {acc_sum / val_df.shape[0]}')
        plot_accuracies(accuracies, elos_players, maia_elos, n_moves=n_boards)
        df_out = pd.DataFrame(np.array([
            np.array([[maia_elo] * len(elos_players[-1]) for maia_elo in maia_elos]).flatten(),
            np.array(elos_players).flatten(),
            np.array(accuracies).flatten(),
                      ]).T,
                     columns=['Elo Maia', 'Elo Players', 'Accuracy'])
        os.makedirs(f'results/maia_replication', exist_ok=True)
        df_out.to_csv(f'results/maia_replication/{maia_elos[0]}_{maia_elos[-1]}_{n_boards}.csv')
