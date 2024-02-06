from project.chess_utils.sunfish import Position, Move, Searcher, render
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
    return best_move, info


def sunfish_move_to_str(move: Move, is_black:bool=False):
    i, j = move.i, move.j
    if is_black:
        i, j = 119 - i, 119 - j
    move_str = render(i) + render(j) + move.prom.lower()
    return move_str

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
    # Assert either normal move or promotion
    assert (len(move) == 4 or len(move) == 6), 'Move must be 4 or 6 (when promoting) chars long'
    i = square2sunfish(move[:2])
    j = square2sunfish(move[2:4])
    prom = move[5] if len(move) > 4 else ''
    return Move(i, j, prom)

# Takes a board object and returns the position
# in the format sunfish uses. Mangler score.
def board2sunfish(board):
    fen = board.fen()

    board_string, to_move, castling, ep, half_move, full_move = fen.split()

    start = '         \n         \n '
    end = '         \n         \n'
    board_string = board_string.replace('/', '\n ')
    board_string = start + board_string + end
    for char in board_string:
        if char in '123456789':
            board_string = board_string.replace(char, int(char) * '.')

    score = 0

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
