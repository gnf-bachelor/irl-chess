from copy import deepcopy

import torch
import chess
import numpy as np
from tqdm import tqdm
from time import time
from copy import copy
from project.chess_utils.sunfish_utils import board2sunfish
from project.chess_utils.sunfish import piece

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


def evaluate_board(board, R, white=False):
    """
    positive if (w + not lower), (not w + lower)
    negative if (w + lower), (not w + not lower)

    :param board:
    :param R:
    :param white:
    :return:
    """
    eval = 0
    for lower in (False, True):
        keys = {val.lower() if lower else val: 0 for val in piece.keys()}
        for char in board.fen():
            if char in keys:
                keys[char] += 1
        pos = np.array([val for val in keys.values()])
        eval += (pos @ R) * (-1 if white == lower else 1)
    return eval


def alpha_beta_search(board,
                      depth,
                      alpha=-np.inf,
                      beta=np.inf,
                      maximize=True,
                      R: np.array = np.zeros(1),
                      evaluation_function=evaluate_board):
    """
    When maximize is True the board must be evaluated from the White
    player's perspective.

    :param board:
    :param depth:
    :param alpha:
    :param beta:
    :param maximize:
    :param R:
    :param evaluation_function:
    :return:
    """
    if depth == 0 or board.is_game_over():
        return evaluation_function(board, R, maximize)

    if maximize:
        max_eval = -np.inf
        for move in board.generate_legal_moves():
            board.push(move)
            eval = alpha_beta_search(board, depth - 1, alpha, beta, False, R=R, evaluation_function=evaluation_function)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = np.inf
        for move in board.generate_legal_moves():
            board.push(move)
            eval = alpha_beta_search(board, alpha=alpha, depth=depth - 1, maximize=True, R=R, evaluation_function=evaluation_function)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval


def get_best_move(board, R, depth=3, timer=False, evaluation_function=evaluate_board, white=True):
    """

    :param board:
    :param R:
    :param depth:
    :param timer:
    :param evaluation_function:
    :param white:               Is it white's turn to make a move
    :return:
    """
    best_move, Q = None, None
    alpha = -np.inf
    moves = tqdm([move for move in board.legal_moves]) if timer else board.legal_moves
    for move in moves:
        board.push(move)
        Q = alpha_beta_search(board, alpha=alpha, depth=depth - 1, maximize=not white, R=R, evaluation_function=evaluation_function)
        board.pop()
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


def get_midgame_boards(df,
                       n_boards,
                       min_elo,
                       max_elo,
                       n_steps=12,
                       sunfish=False):
    """
    Using chess.Board() as the moves are currently in that format.
    Needs a DataFrame with 'Moves', 'WhiteElo' and 'BlackElo'
    columns. The midgame is defined by the n_steps.
    The timer is inaccurate and shows an upper bound
    :param df:
    :param n_boards:
    :param min_elo:
    :param max_elo:
    :param n_steps:
    :return:
    """
    boards, moves = [], []

    for moveset, elo_w, elo_b in tqdm(df[['Moves', 'WhiteElo', 'BlackElo']].values):
        board = chess.Board()
        moveset_split = moveset.split(',')[:-2]
        if len(moveset_split) > n_steps and (min_elo <= int(elo_w) <= max_elo) and (min_elo <= int(elo_b) <= max_elo):
            try:
                for move in moveset_split[:-1]:
                    board.push_san(move)
                board.push_san(moveset_split[-1])
                board.pop()
                moves.append(moveset_split[-1])
                if sunfish:
                    boards.append(board2sunfish(board))
                else:
                    boards.append(copy(board))
            except chess.InvalidMoveError:
                pass
        if len(boards) == n_boards:
            break
    return boards, moves


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


def prob_dist(R, energy, alpha, prior=lambda R: 1):
    prob = np.exp(alpha * energy) * prior(R)
    return prob


def policy_walk(R, states, moves, delta=1e-3, epochs=10, depth=3, alpha=2e-2):
    for epoch in tqdm(range(epochs)):
        add = np.random.rand(R.shape[0]).astype(R.dtype) * (delta / 2)
        R_ = R + add
        Q_moves = np.zeros(len(states))
        Q_policy = np.zeros(len(states))
        i = 0
        energy_new, energy_old = 0, 0
        for state, move in tqdm(zip(states, moves), total=len(states)):
            state.push_san(move)
            _, Q_old = get_best_move(board=state, R=R, depth=depth - 1, white=state.turn)
            _, Q_new = get_best_move(board=state, R=R_, depth=depth - 1, white=state.turn)
            state.pop()
            # _, Q_old_energy = get_best_move(board=state, R=R, depth=depth)

            Q_moves[i] = Q_old
            Q_policy[i] = Q_new

            energy_old += Q_old
            energy_new += Q_new

            i += 1
            prob = min(1, prob_dist(R_, energy_new, alpha=alpha) / prob_dist(R_, energy_old, alpha=alpha))
            if np.sum(Q_policy < Q_moves):
                if np.random.rand(1).item() < prob:
                    R = R_
    return R