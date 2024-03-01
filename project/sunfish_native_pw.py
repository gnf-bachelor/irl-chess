import os
from os.path import join
import copy
import chess
import chess.pgn
import chess.svg
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from project import Searcher, Position, initial, sunfish_move, sunfish_move_to_str, pst, pst_only, piece
from project.chess_utils.sunfish_utils import board2sunfish, sunfish_move_to_str, render, sunfish2board

def get_board_after_n(game, n):
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        board.push(move)
        if i == n:
            break
    return board

def get_board_last(tp_move, init_pos):
    pos_ = init_pos
    while pos_ in tp_move.keys():
        move = tp_move[pos_]
        pos_ = pos_.move(move)
    return pos_

# Assuming white, R is array of piece values
def eval_pos(board, R=None):
    pos = board2sunfish(board, 0)
    pieces = 'PNBRQK'
    if R is not None:
        piece_dict = {p: R[i] for i, p in enumerate(pieces)}
    else:
        piece_dict = piece
    eval = 0
    for row in range(20,100,10):
        for square in range(1 + row,9 + row):
            p = pos.board[square]
            if p == '.':
                continue
            if p.islower():
                p = p.upper()
                eval -= piece_dict[p] + pst[p][119-square]
            else:
                eval += piece_dict[p] + pst[p][square]
    return eval

def get_new_pst(R):
    assert len(R) == 6
    pieces = 'PNBRQK'
    piece_new = {p: val for p, val in list(zip(pieces, R))}
    pst_new = copy.deepcopy(pst_only)
    for k, table in pst_only.items():
        padrow = lambda row: (0,) + tuple(x + piece_new[k] for x in row) + (0,)
        pst_new[k] = sum((padrow(table[i * 8: i * 8 + 8]) for i in range(8)), ())
        pst_new[k] = (0,) * 20 + pst_new[k] + (0,) * 20
    return pst_new

def sunfish_move_mod(state, pst, time_limit, only_move=False):
    searcher = Searcher(pst)
    if only_move:
        return sunfish_move(searcher, [state], time_limit=time_limit)[0]
    return sunfish_move(searcher, [state], time_limit=time_limit)

def plot_R(Rs):
    Rs = np.array(Rs)
    plt.plot(Rs[:, :-1],)
    plt.title('Piece values by epoch')
    plt.legend(list('PNBRQ'))
    plt.show()

# actions = []
# for state in tqdm(states):
#     move = sunfish_move_mod(state, pst_new,time_limit,True)
#     actions.append(move)

if __name__ == '__main__':
    if os.getcwd().split('\\')[-1] != 'irl-chess':
        os.chdir('../')

    pgn = open("data/lichess_db_standard_rated_2014-09.pgn/lichess_db_standard_rated_2014-09.pgn")
    games = []
    for i in range(1000):
        games.append(chess.pgn.read_game(pgn))

    n_games = 500
    states_boards = [get_board_after_n(game, 15) for game in games[:n_games]]
    states = [board2sunfish(board, eval_pos(board)) for board in states_boards]

    epochs = 400
    time_limit = 0.1
    step = 20
    decay = 0.95
    best_accs = [0]
    Rs = []
    with Parallel(n_jobs=-2) as parallel:
        print('Getting true moves\n', '-' * 20)
        actions_true = parallel(delayed(sunfish_move_mod)(state, pst, time_limit, True)
                                for state in tqdm(states))
        R = np.array([100, 280, 100, 100, 929, 60000])
        for i in range(epochs):
            print(f'Epoch {i + 1}\n', '-' * 20)
            R_new = R + np.pad(np.random.choice([-step, step], 2), 2)
            pst_new = get_new_pst(R_new)
            states = [board2sunfish(board, eval_pos(board, R_new)) for board in states_boards]
            actions_new = parallel(delayed(sunfish_move_mod)(state, pst_new, time_limit, True)
                                   for state in tqdm(states))

            acc = sum([a == a_new for a, a_new in list(zip(actions_true, actions_new))])/n_games
            if acc >= best_accs[-1]:
                R = R_new
                best_accs.append(acc)
            Rs.append(R)

            if i % 10 == 0 and i != 0:
                plot_R(Rs)

            if i % 30 == 0 and i != 0:
                step *= decay

            print(f'Current accuracy: {acc}')
    plot_R(Rs)