import chess
import numpy as np
from tqdm import tqdm
import copy
from time import time

from irl_chess.chess_utils.sunfish import Position, Move, Searcher, render, pst, pst_only, pst_only_padded, piece

def sunfish_move(state, pst, time_limit, move_only=False, max_depth=1000, run_at_least=1, min_depth=1, return_best_board_found_tuple = False):
    """
    Given a state, p-square table and time limit,
    return the sunfish move.
    :param state:
    :param pst:
    :param time_limit:
    :return:
    """
    searcher = Searcher(pst, max_depth=max_depth)
    start = time()
    best_move = None
    count = 0
    count_gamma = 0
    for depth, gamma, score, move in searcher.search([state]):
        count += 1
        if score >= gamma:
            best_move = move
            count_gamma += 1
        if time() - start > time_limit and count_gamma >= 1 and (count >= run_at_least) and depth >= min_depth:
            break
    if best_move is None:
        print(f"best move is: {best_move} and count is {count}")
        print(state.board)
    assert best_move is not None, f"No best move found, this probably means an invalid position was passed to the \
                                   searcher"
    if move_only: # Should clean this up such that it returns (best_move, info) where info is dictionary with all the auxillary information. 
        return best_move
    elif return_best_board_found_tuple:
        return best_move, searcher.best_moves, searcher.move_dict, sunfish_best_board(state, pst, searcher.tp_move)
    else: 
        return best_move, searcher.best_moves, searcher.move_dict

def sunfish_best_board(state, pst, tp_move: dict):
    state_depth = 0
    while state in tp_move:
        best_move = tp_move[state]
        next_state = state.move(best_move, pst)
        state = next_state
        state_depth += 1
    assert state_depth > 0
    # print(state_depth)
    score = eval_pos_pst(state, pst)*(-1 if state_depth % 2 else 1) # Keep trach of whether it is white's or black's turn
    return state, score, bool(state_depth % 2) # state, score, opposite players turn or not

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

    wc = ('Q' in castling, 'K' in castling)
    bc = ('q' in castling, 'k' in castling)

    if ep == '-':
        ep = 0
    else:
        ep = square2sunfish(ep) # Check later. 

    kp = 0

    # Reverse if black to move
    if to_move == 'b':
        return Position(board_string[::-1].swapcase(), -score, bc, wc,
                        119 - ep if ep else 0,
                        119 - kp if kp else 0)

    return Position(board_string, score, wc, bc, ep, kp)

def get_new_pst(RP, Rpst = None):
    # Get a new set of piece square tables (pst), but with the pieces weighed by the RP values 
    # and the pst values weighed by the Rpst values.
    assert len(RP) == 6
    if Rpst is not None: assert len(Rpst) == 6
    else:
        Rpst = [1, 1, 1, 1, 1, 1]
    pieces = 'PNBRQK'
    piece_new = {p: val for p, val in list(zip(pieces, RP))}
    pst_new = copy.deepcopy(pst_only)
    for j, (k, table) in enumerate(pst_only.items()):
        padrow = lambda row: (0,) + tuple(x*Rpst[j] + piece_new[k] for x in row) + (0,)
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
def eval_pos(board, RP=None, Rpst=None, RH=None):
    if isinstance(board, chess.Board):
        #assert board.turn == True # Assume it is white's turn. # Fish all of sunfishes rotation mistakes later. 
        pos = board2sunfish(board, 0)
    elif isinstance(board, Position):
        pos = board
    else:
        raise TypeError
    pieces = 'PNBRQK'
    if RP is not None:
        piece_dict = {p: RP[i] for i, p in enumerate(pieces)} 
    else:
        piece_dict = piece
    if Rpst is not None:
        pst_weight_dict = {p: Rpst[i] for i, p in enumerate(pieces)} 
    else:
        pst_weight_dict = {p: 1 for p in pieces}   
    eval = 0
    # pst is of size 120 12x10 (rows, columns) padded.
    for row in range(20, 100, 10):  
        for col in range(1, 9):
            # White square = (row + col)
            p = pos.board[(row + col)]
            if p == '.':
                continue
            if p.islower(): # if black piece. Mirror baord. 
                p = p.upper()
                eval -= piece_dict[p] + pst_weight_dict[p]*pst_only_padded[p][110 - row + col]
            else: # else if white piece
                eval += piece_dict[p] + pst_weight_dict[p]*pst_only_padded[p][(row + col)]
    return eval

def eval_pos_pst(board, pst):
    if isinstance(board, chess.Board):
        #assert board.turn == True # Assume it is white's turn. # Fish all of sunfishes rotation mistakes later. 
        pos = board2sunfish(board, 0)
    elif isinstance(board, Position):
        pos = board
    else:
        raise TypeError
    eval = 0
    # pst is of size 120 12x10 (rows, columns) padded.
    for row in range(20, 100, 10):  
        for col in range(1, 9):
            # White square = (row + col)
            p = pos.board[(row + col)]
            if p == '.':
                continue
            if p.islower(): # if black piece. Mirror baord. 
                p = p.upper()
                eval -= pst[p][110 - row + col]
            else: # else if white piece
                eval += pst[p][(row + col)]
    return eval

# check sunfish moves same color as player
def check_moved_same_color(sunfish_boards, player_moves_sunfish, actions_new):
    for k, pos in enumerate(sunfish_boards):
        player_move_square = player_moves_sunfish[k].i
        sunfish_move_square = actions_new[k].i
        move_ok = pos.board[player_move_square].isupper() == pos.board[sunfish_move_square].isupper()
        assert move_ok, 'Wrong color piece moved by sunfish'

def moves_and_Q_from_result(results, states):
    moves, _, Q_new_dicts = [], [], {}
    for state, (move, best_moves, move_dict) in list(zip(states, results)):
        assert move is not None, "Move should never be None here"
        moves.append(sunfish_move_to_str(move))
        Q_new_dicts[state] = {sunfish_move_to_str(move): scores[-1] for move, scores in move_dict.items()
                                if move is not None}
    return moves, Q_new_dicts