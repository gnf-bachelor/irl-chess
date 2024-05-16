import chess
import numpy as np
from tqdm import tqdm
import copy

from irl_chess.chess_utils.sunfish import Position, Move, Searcher, render, pst, piece
from irl_chess.chess_utils.sunfish import pst_only


def sunfish_move_to_str(move: Move, is_black:bool):
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
def str_to_sunfish_move(move, flip):
    if not isinstance(move, str):
        move = move.uci()
    # Assert either normal move or promotion
    assert (len(move) == 4 or len(move) == 5), 'Move must be 4 or 5 chars long'
    i = square2sunfish(move[:2])
    j = square2sunfish(move[2:4])
    if flip:
        i = 119 - i
        j = 119 - j
    prom = move[4] if len(move) > 4 else ''
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


# Assuming white, R is array of piece values
def eval_pos(board, R=None):
    pos = board2sunfish(board, 0)
    pieces = 'PNBRQK'
    if R is not None:
        piece_dict = {p: R[i] for i, p in enumerate(pieces)}
    else:
        piece_dict = piece
    eval = 0
    for row in range(20, 100, 10):
        for square in range(1 + row, 9 + row):
            p = pos.board[square]
            if p == '.':
                continue
            if p.islower():
                p = p.upper()
                eval -= piece_dict[p] + pst[p][119 - square]
            else:
                eval += piece_dict[p] + pst[p][square]
    return eval

