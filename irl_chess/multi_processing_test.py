import os
from os.path import join
import chess
import chess.pgn
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from irl_chess import Searcher, Position, initial, sunfish_move, sunfish_move_to_str, alpha_beta_search, get_best_move

if os.getcwd().split('\\')[-1] != 'irl-chess':
    os.chdir('../')

pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
games = []
for i in range(1000):
    games.append(chess.pgn.read_game(pgn))

boards = []
for game in games[:20]:
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i == 12:
            break
        board.push(move)
    boards.append(board)
# TEST


Q = []
print('Running without multiprocessing')
for board in tqdm(boards):
    q = alpha_beta_search(board, depth=4, R=np.ones(6))
    Q.append(q)
print(Q[-1])

print('Running with multiprocessing')
Q_p = Parallel(n_jobs=-2)(delayed(alpha_beta_search)(board, depth=4, R=np.ones(6)) for board in tqdm(boards))
print(Q_p[-1])





