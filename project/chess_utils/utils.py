from copy import deepcopy
from os.path import join

import pandas as pd
import torch
import chess
import numpy as np
from tqdm import tqdm
from time import time
from copy import copy
from project.chess_utils.sunfish_utils import board2sunfish
from project.chess_utils.sunfish import piece, pst
from scipy.special import softmax

material_dict = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # The value for the king is typically set to 0 in material evaluation
}

# pst eval added for black
pst_only = pst
for piece in pst.keys():
    piece_array = -np.array(pst[piece]).reshape((8, 8))[np.arange(7, -1, -1)]
    pst_only[piece.lower()] = tuple(piece_array.reshape((-1)))

def set_board(moves: list[str]):
    board = chess.Board()
    for move in moves:
        board.push_san(move)
    return board

# Evaluates the piece positions according to sunfish
def eval_pst_only(board):
    s = str(board()).replace(' ', '').replace('\n', '')
    eval = 0
    for i, char in enumerate(s):
        if char == '.':
            continue
        else:
            eval += pst[char][i]
    return eval

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
        return evaluation_function(board, R, maximize), board

    if maximize:
        max_eval = -np.inf
        for move in board.generate_legal_moves():
            board.push(move)
            eval, board_last = alpha_beta_search(board, depth - 1, alpha, beta, False, R=R, evaluation_function=evaluation_function)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval, board_last
    else:
        min_eval = np.inf
        for move in board.generate_legal_moves():
            board.push(move)
            eval, board_last = alpha_beta_search(board, alpha=alpha, depth=depth - 1, maximize=True, R=R,
                                     evaluation_function=evaluation_function)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval, board_last


def get_best_move(board, R, depth=3, timer=False, evaluation_function=evaluate_board, white=True, san=False):
    """

    :param board:
    :param R:
    :param depth:
    :param timer:
    :param evaluation_function:
    :param white:               Is it white's turn to make a move?
    :return:
    """
    best_move, Q = None, None
    alpha = -np.inf
    moves = tqdm([move for move in board.legal_moves]) if timer else board.legal_moves
    for move in moves:
        board.push(move)
        Q, _ = alpha_beta_search(board, alpha=alpha, depth=depth - 1, maximize=not white, R=R,
                              evaluation_function=evaluation_function)
        board.pop()
        if Q > alpha:
            alpha = Q
            best_move = board.san(move) if san else move
    return best_move, alpha


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
                       sunfish=False,
                       move_translation=lambda move: move):
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
    :param move_translation: can be a function that ensures moves are of the desired format
    :return:
    """
    boards, moves = [], []

    for moveset, elo_w, elo_b in tqdm(df[['Moves', 'WhiteElo', 'BlackElo']].values, desc='Searching for boards'):
        board = chess.Board()
        moveset_split = moveset.split(',')[:-2]
        if len(moveset_split) > n_steps and (min_elo <= int(elo_w) <= max_elo) and (min_elo <= int(elo_b) <= max_elo):
            try:
                for i, move in enumerate(moveset_split[:-1]):
                    board.push_san(move)
                board.push_san(moveset_split[-1])
                if len([el for el in board.generate_legal_moves()]):
                    board.pop()
                    moves.append(move_translation(moveset_split[-1]))

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


def softmax_choice(x):
    """
    Returns an index based on the softmax of x and add 1 to ensure 
    the depth is never 0
    :param x: 
    :return: 
    """
    choice = np.random.choice(np.arange(len(x)), p=softmax(x))
    return choice + 1


def log_prob_dist(R, energy, alpha, prior=lambda R: 1):
    log_prob = alpha * energy + np.log(prior(R))
    return log_prob


def policy_walk(R, boards, moves, delta=1e-3, epochs=10, depth=3, alpha=2e-2, permute_all=True, save_every=None, save_path=None, san=True):
    """ Policy walk algorithm over given class of reward functions.
    Iterates over the initial reward function by perterbing each dimension uniformly and then
    accepting the new reward function with probability proportional to how much better they explain the given trajectories. 

    Args:
        R (_type_): The reward function (heuristic for statically evaluation a board).
        boards (_type_): list of chess.Move() objects
        moves (_type_): _description_
        delta (_type_, optional): _description_. Defaults to 1e-3.
        epochs (int, optional): _description_. Defaults to 10.
        depth (int, optional): _description_. Defaults to 3.
        alpha (_type_, optional): _description_. Defaults to 2e-2.

    Returns:
        _type_: _description_
    """
    for epoch in tqdm(range(epochs), desc='Iterating over epochs'):
        i = 0
        Q_newR_O = np.zeros(len(boards))
        Q_newR_DO = np.zeros(len(boards))
        Q_boards_oldR_DO_list = []
        energy_newR_DO, energy_oldR_DO = 0, 0

        R_ = R
        if permute_all:
            add = np.random.uniform(low=-delta, high=delta, size=R.shape[0] - 1).astype(R.dtype)
            R_[1:] += add
        else:
            choice = np.random.choice(np.arange(len(R_)))
            R_[choice] += np.random.rand(1).item() * (delta / 2)

        for board, move in tqdm(zip(boards, moves), total=len(boards), desc='Policy walking over reward functions'):

            board.push_san(move) if san else board.push(move)
            # First we get the board from the state/action pair seen in the data using the old weights
            if len(Q_boards_oldR_DO_list):
                Q_oldR_DO_policy, board_old = Q_boards_oldR_DO_list[i]
            else:
                Q_oldR_DO_policy, board_old, = alpha_beta_search(board=board, R=R, depth=depth-1, maximize=board.turn)
            # Then we evaluate the found board using the new weights
            Q_newR_DO_policy = evaluate_board(board=board_old, R=R_, white=board_old.turn)     # Q^pi(s,a,R_)
            board.pop()
            # Finally we calculate the Q-value of the old policy on the state without the original move
            _, board_old_, = alpha_beta_search(board=board, R=R, depth=depth, maximize=board.turn)
            Q_newR_O_policy = evaluate_board(board=board, R=R_, white=board_old_.turn)    # Q^pi(s,pi(s),R_)

            Q_newR_O[i] = Q_newR_O_policy     # Q^pi(s,pi(s),R_)
            Q_newR_DO[i] = Q_newR_DO_policy   # Q^pi(s,a,R_)

            energy_oldR_DO += Q_oldR_DO_policy
            energy_newR_DO += Q_newR_DO_policy
            i += 1

        if np.sum(Q_newR_DO < Q_newR_O):
            energy_newR_DN = 0
            for board, move in tqdm(zip(boards, moves), total=len(boards), desc='Calculating Q-values for new Policy'):
                Q_newR_DN_policy, board_newR_DN = alpha_beta_search(board=board, R=R_, depth=depth, maximize=board.turn)
                Q_boards_oldR_DO_list.append((Q_newR_DN_policy, board_newR_DN))
                energy_newR_DN += Q_newR_DN_policy
            log_prob = min(0, log_prob_dist(R_, energy_newR_DN, alpha=alpha) - log_prob_dist(R, energy_oldR_DO, alpha=alpha))
        else:
            log_prob = min(0, log_prob_dist(R_, energy_newR_DO, alpha=alpha) - log_prob_dist(R, energy_oldR_DO, alpha=alpha))

        p = np.random.rand(1).item()
        if log_prob > -1e7 and p < np.exp(log_prob):
            R = R_
        if save_every is not None and epoch % save_every == 0:
            pd.DataFrame(R_.reshape((-1, 1)), columns=['Result']).to_csv(join(save_path, f'{epoch}.csv'), index=False)

    return R


def policy_walk_depth(R, boards, moves, delta=1e-3, epochs=10, depth_max=3, alpha=2e-2, time_max=np.inf,
                      timer_moves=False):
    """ Policy walk algorithm over the depth of the search.
    Begins with a uniform distribution over search depths and iterates by perterbing each dimension uniformly and then
    accepting the new softmax distribution of search depths with probability proportional to how much better they explain the given trajectories. 

    Args:
        R (_type_): The reward function (heuristic for statically evaluation a board).
        boards (_type_): _description_
        moves (_type_): _description_
        delta (_type_, optional): _description_. Defaults to 1e-3.
        epochs (int, optional): _description_. Defaults to 10.
        depth_max (int, optional): _description_. Defaults to 3.
        alpha (_type_, optional): _description_. Defaults to 2e-2.
        time_max (_type_, optional): _description_. Defaults to np.inf.
        timer_moves (bool, optional): _description_. Defaults to False.

    Returns:
        ndarray : final array of search depth probability distribution pre-softmax. Indexes signify depth starting from 1. 
    """
    depth_dist = np.ones(depth_max)
    start = time()
    for epoch in tqdm(range(epochs)):
        Q_moves = np.zeros(len(boards))
        Q_policy = np.zeros(len(boards))
        i = 0
        energy_new, energy_old = 0, 0
        for board, move in tqdm(zip(boards, moves), total=len(boards)):
            add = np.random.uniform(low=-delta, high=delta, size=depth_dist.shape[0]).astype(
                depth_dist.dtype)  # * (delta / 2)
            depth_dist_ = depth_dist + add

            board.push(move)
            depth1 = softmax_choice(depth_dist)
            depth2 = softmax_choice(depth_dist_)
            _, Q_old = get_best_move(board=board, R=R, depth=depth1, timer=timer_moves, white=board.turn)
            _, Q_new = get_best_move(board=board, R=R, depth=depth2, timer=timer_moves, white=board.turn)
            if Q_new is None or Q_old is None:
                continue
            board.pop()

            Q_moves[i] = Q_old
            Q_policy[i] = Q_new

            energy_old += Q_old
            energy_new += Q_new

            i += 1
            # prob = min(1, prob_dist(depth_dist_, energy_new, alpha=alpha)/prob_dist(depth_dist, energy_old, alpha=alpha))

            log_prob = min(0,
                           log_prob_dist(depth_dist_, energy_new, alpha=alpha) - log_prob_dist(depth_dist, energy_old,
                                                                                               alpha=alpha))

            if np.sum(Q_policy < Q_moves):
                if log_prob > -1e7 and np.random.rand(1).item() < np.exp(log_prob):
                    depth_dist = copy(depth_dist_)
            if start - time() > time_max:
                return depth_dist
    return depth_dist
