from copy import deepcopy

import torch
import chess
import numpy as np
from tqdm import tqdm
from time import time


material_dict = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # The value for the king is typically set to 0 in material evaluation
}


def set_board(moves: list[str]):
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board


def calculate_heuristics(board: chess.Board, tensor=False):
    board_arr = board_to_array(board, material_dict=material_dict, tensor=tensor)

    # material_white = np.sum(board_arr[board_arr > 0])
    # material_black = np.abs(np.sum(board_arr[board_arr < 0]))
    pieces_white = [np.sum(board_arr == i) if not tensor else torch.sum(board_arr == i) for i in range(1, 7)]
    pieces_black = [- np.sum(board_arr == i) if not tensor else torch.sum(board_arr == i) for i in range(-6, 0)]
    # check = board.is_check()
    # checkmate = board.is_checkmate()
    out = (*pieces_white, *pieces_black, ) # check, checkmate)
    return torch.tensor(out, dtype=torch.float) if tensor else np.array(out, dtype=float)


def evaluate(board: chess.Board, R):
    return calculate_heuristics(board, tensor=False) @ R


def alpha_beta_search(board,
                      depth,
                      alpha=-np.inf,
                      beta=np.inf,
                      maximize=True,
                      R: np.array = np.zeros(1),
                      evaluation_function=evaluate):
    if depth == 0 or board.is_game_over():
        return evaluation_function(board, R)

    if maximize:
        max_eval = -np.inf
        for move in board.generate_legal_moves():
            board_ = deepcopy(board)
            board_.push(move)
            eval = alpha_beta_search(board_, depth - 1, alpha, beta, False, R=R)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = np.inf
        for move in board.generate_legal_moves():
            board_ = deepcopy(board)
            board_.push(move)
            eval = alpha_beta_search(board_, depth - 1, alpha, beta, True, R=-R)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


def get_best_move(board, R, depth=3, timer=False, evaluation_function=evaluate):
    best_move, Q = None, None
    alpha = -np.inf
    moves = tqdm([move for move in board.legal_moves]) if timer else board.legal_moves
    for move in moves:
        board_ = deepcopy(board)
        board_.push(move)
        Q = alpha_beta_search(board_,
                              alpha=alpha,
                              depth=depth-1,
                              maximize=True,
                              R=R,
                              evaluation_function=evaluation_function)
        if Q > alpha:
            alpha = Q
            best_move = move
    return best_move, Q


def get_board_arrays(game_moves):
    board = chess.Board()
    positions = []

    for move in game_moves:
        board.push_san(move)
        positions.append(board_to_array(board))

    return positions


def board_to_array(board, material_dict=None, tensor=False, dtype=np.int8):
    if material_dict is None:
        material_dict = {i: i for i in range(1, 7)}
    arr = np.zeros(64, dtype=dtype)
    for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                arr[square] = material_dict[piece.piece_type] * (1 if piece.color else -1)
    # arr = arr.reshape((8, 8))     # Need a reason to reshape
    return arr if not tensor else torch.tensor(arr)


def depth_first_search(starting_board: chess.Board,
                       true_move: str,
                       weights: np.array = np.ones(1),
                       depth: int = 2,
                       heuristic_function=None):
    if heuristic_function is None:
        def heuristic_function(board: chess.Board):
            return np.ones_like(weights)
    # depth refers to depth of moves by moving player
    boards_seen = [[starting_board]] + [[] for _ in range(depth * 2)]
    boards_not_seen = [[starting_board]] + [[] for _ in range(depth * 2)]

    for i in tqdm(range(depth * 2)):
        for board in boards_not_seen[i]:
            for move in board.legal_moves:
                san_move = board.san(move)
                board.push(move)
                if san_move != true_move:
                    boards_not_seen[i + 1].append(deepcopy(board))
                board.pop()
        for board in boards_seen[i]:
            for move in board.legal_moves:
                board.push(move)
                boards_seen[i + 1].append(deepcopy(board))
                board.pop()

    return boards_not_seen, boards_seen
