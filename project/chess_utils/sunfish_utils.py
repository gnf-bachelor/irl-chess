import chess
import numpy as np
from tqdm import tqdm

from project.chess_utils.sunfish import Position, Move, Searcher, render, pst
from time import time

def sunfish_move(searcher: Searcher, hist: list[Position], time_limit:float=1., ) -> tuple[Move, dict]:
    """
    Given a sunfish searcher, the game history so far
    and a time_limit in seconds, return the move that
    in the given time was found to have the best score.
    Also returns a dictionary with the info from the
    search. If there is no best move found it (this
    should in theory be impossible) it crashes.
    """
    start = time()
    best_move = None
    for depth, gamma, score, move in searcher.search(hist):
        if score >= gamma:
            best_move = move
        if best_move and time() - start > time_limit:
            break

    info = {'depth': depth,
            'gamma': gamma,
            'score': score,
            'nodes': searcher.nodes}
    return best_move, info, searcher.tp_move, searcher.tp_score


def sunfish_move_to_str(move: Move, is_black:bool=False):
    i, j = move.i, move.j
    if is_black:
        i, j = 119 - i, 119 - j
    move_str = render(i) + render(j) + move.prom.lower()
    return move_str


# def get_best_move_sunfish(board, R, depth=3, timer=False):
#     best_move, Q = None, None
#     alpha = -np.inf
#     moves = tqdm([move for move in board.legal_moves]) if timer else board.legal_moves
#     for move in moves:
#         board.push(move)
#         Q = alpha_beta_search(board, alpha=alpha, depth=depth-1, maximize=True, R=R)
#         board.pop()
#         if Q > alpha:
#             alpha = Q
#             best_move = move
#     return best_move, Q

# takes squares in the form 'a2', 'g3' etc. and returns
# the number used to represent it in sunfish.
def square2sunfish(square):
    assert len(square) == 2
    col, row = list(square)
    row = int(row) - 1
    col = 'abcdefgh'.find(col.lower()) + 1
    sf_square = 90 - row * 10 + col
    return sf_square

# Normal moves assumed to be in format 'e4e5', promotions
# assumed to be eg. 'e7e8=Q'
def str_to_sunfish_move(move):
    if not isinstance(move, str):
        move = move.uci()
    # Assert either normal move or promotion
    assert (len(move) == 4 or len(move) == 6), 'Move must be 4 or 6 chars long'
    i = square2sunfish(move[:2])
    j = square2sunfish(move[2:4])
    prom = move[5] if len(move) > 4 else ''
    return Move(i, j, prom)

# Takes a board object and returns the position
# in the format sunfish uses. Mangler score.
def board2sunfish(board, score):
    fen = board.fen()

    board_string, to_move, castling, ep, half_move, full_move = fen.split()

    start = '\n         \n         \n'
    end = ' \n         \n         '
    board_string = board_string.replace('/', ' \n')
    board_string = start + board_string + end
    for char in board_string:
        if char in '123456789':
            board_string = board_string.replace(char, int(char) * '.')

    score = score

    wc = ('Q' in castling, 'K' in castling)
    bc = ('q' in castling, 'k' in castling)

    if ep == '-':
        ep = 0
    else:
        ep = square2sunfish(ep)

    kp = 0

    # Reverse if black to move
    if to_move == 'b':
        return Position(board_string[::-1].swapcase(), -score, bc, wc,
                        119 - ep if ep else 0,
                        119 - kp if kp else 0)

    return Position(board_string, score, wc, bc, ep, kp)

def sunfish2board(pos: Position):
    pos_string = pos.board[21:-20].replace(' \n', '/')
    for i in range(8, 0, -1):
        pos_string = pos_string.replace('.' * i, str(i))
    to_move = 'w'
    castling = ''
    for i, condition in enumerate(pos.wc + pos.bc):
        if condition:
            castling += 'KQkq'[i]
    if pos.ep == 0:
        ep = '-'
    else:
        ep = render(ep)
    fen = str(' '.join([pos_string[:-1], to_move, castling, ep, '0 0']))
    board = chess.Board()
    board.set_fen(fen)
    return board

