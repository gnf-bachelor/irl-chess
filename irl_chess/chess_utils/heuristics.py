import numpy as np

# Get a map of the pawns on the board with no pawn represented
# by a zero, a white pawn by 1, and a black pawn by -1.
def get_pawn_array(board):
    board_str = str(board).replace(' ', '').replace('\n', '')
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


def pawn_eval(board, test=False):
    weight = {'iso': -1, 'dob': -1, 'pas': 1}

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

    return sum([isolated, double, passed])