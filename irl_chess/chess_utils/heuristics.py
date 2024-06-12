import numpy as np
from irl_chess.chess_utils.sunfish_utils import board2sunfish
import chess

# Get a map of the pawns on the board with no pawn represented
# by a zero, a white pawn by 1, and a black pawn by -1.
def get_pawn_array(board):
    board_str = board2sunfish(board, 0).board
    board_str = board_str.replace(' ', '').replace('\n', '')
    pawns = {'p': -1, 'P': 1}
    pawn_array = [pawns[char] if char in 'Pp' else 0 for char in board_str]
    pawn_array = np.array(pawn_array).reshape((8, 8))
    return np.pad(pawn_array, 1)


# Count passed pawns
def count_passed_pawns(pawn_array):
    white_pawn_squares = np.array(np.where(pawn_array == 1)).T
    black_pawns = (pawn_array == -1)
    # count the number of pawns that oppose each white pawn.
    opposite_pawns = [(black_pawns[:i, j - 1:j + 2] == 1).sum() for i, j in white_pawn_squares]
    return sum([1 if op == 0 else 0 for op in opposite_pawns])


def count_isolated_pawns(file_count):
    return sum([file_count[[i - 1, i + 1]].sum() == 0
                if pawns else 0
                for i, pawns in enumerate(file_count)])


def pawn_eval(board, pawn_array = None, test=False):
    weight = {'iso': -1, 'dob': -1, 'pas': 2}

    if pawn_array is None: 
        pawn_array = get_pawn_array(board)
    file_count_w = (pawn_array.T == 1).sum(axis=1)
    file_count_b = (pawn_array.T == -1).sum(axis=1)

    double = weight['dob'] * (sum(file_count_w == 2) - sum(file_count_b == 2))
    isolated = weight['iso'] * (count_isolated_pawns(file_count_w) -
                                count_isolated_pawns(file_count_b))

    passed_w = count_passed_pawns(pawn_array)
    passed_b = count_passed_pawns(np.rot90(pawn_array * (-1), 2))
    passed = weight['pas'] * (passed_w - passed_b)

    if test:
        return isolated, double, passed

    return sum([isolated, double, passed])*20  # 20 seems like a reasonable weighting.


def king_safety(board, pawn_array = None):
    if pawn_array is None: 
        pawn_array = get_pawn_array(board)
    king_square_b = board.king(False)
    i_b, j_b = 8 - chess.square_rank(king_square_b), chess.square_file(king_square_b) + 1

    king_square_w = board.king(True)
    i_w, j_w = 8 - chess.square_rank(king_square_w), chess.square_file(king_square_w) + 1

    center_files = [4, 5]
    center_penalty = 2 * ((j_b in center_files) - (j_w in center_files))

    pawn_shield_w = pawn_array[i_w - 2:i_w, j_w - 1:j_w + 2]
    pawn_shield_b = pawn_array[i_b + 1:i_b + 3, j_b - 1:j_b + 2][::-1]

    shield_weights = np.array([[1, 1, 1], [2, 2, 2]])

    return (np.sum(pawn_shield_w * shield_weights) + np.sum(pawn_shield_b * shield_weights) + center_penalty)*10 
    # 10 seems like a reasonable weighting.

def piece_activity(board):
    # Count legal moves for white
    board.turn = chess.WHITE
    num_legal_moves_white = len(list(board.legal_moves))
    
    # Count legal moves for black
    board.turn = chess.BLACK
    num_legal_moves_black = len(list(board.legal_moves))
    board.turn # legal moves for who? 
    return (np.log(num_legal_moves_white+1) - np.log(num_legal_moves_black))*50  # 100 seems like a reasonable weighting.
    # We aim for weightings on the scale of the pst values, so about an average of 50
