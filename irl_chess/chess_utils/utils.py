from copy import deepcopy
import os
import chess
import numpy as np
from tqdm import tqdm
from copy import copy
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos
from scipy.special import softmax

def perturb_reward(RP, config_data, Rpst = None, RH = None, epoch = None):
    P_permute_idxs = config_data['P_permute_idxs']
    pst_permute_idxs = config_data['pst_permute_idxs']
    H_permute_idxs = config_data['H_permute_idxs']
    delta = config_data['delta']
    delta_pst = delta/200 # Seems reasonable. Prioritize having only one delta parameter.
    delta_H = delta/200 # Seems reasonable too
    noise_distribution = config_data['noise_distribution']
    permute_how_many = config_data.get('permute_how_many', -1) # Default (-1) is to permute all.
    if noise_distribution is None:
        noise_distribution = "uniform"
    P_permute_idxs, pst_permute_idxs, H_permute_idxs = permute_indices(P_permute_idxs, pst_permute_idxs, H_permute_idxs, permute_how_many)
    match noise_distribution:
        case "uniform":
            gen_noise = lambda delta, idxs: np.random.uniform(-delta, delta, size=idxs.shape)
        case "step":
            gen_noise = lambda delta, idxs: np.random.choice([-delta, delta], size=idxs.shape)
        case "gaussian" | "normal":
            gen_noise = lambda delta, idxs: np.random.normal(0, delta, size=idxs.shape)
        case _:
            raise ValueError(f"Unknown noise distribution {noise_distribution}")

    noise_P = gen_noise(delta, P_permute_idxs)
    noise_pst = gen_noise(delta_pst, pst_permute_idxs)
    noise_H = gen_noise(delta_H, H_permute_idxs)
    RP_new = RP.copy()
    # Add noise to the specified indices
    if len(P_permute_idxs) > 0:
        RP_new[P_permute_idxs] += noise_P
    Rpst_new = None
    if Rpst is not None:
        Rpst_new = Rpst.copy()
        if len(pst_permute_idxs) > 0:
            Rpst_new[pst_permute_idxs] += noise_pst
    RH_new = None
    if RH is not None:  
        RH_new = RH.copy()
        if len(H_permute_idxs) > 0:
            RH_new[H_permute_idxs] += noise_H
    if epoch is not None: noise_decay(config_data, epoch)
    assert_RP_returns(RP, RP_new, Rpst, Rpst_new, RH, RH_new)
    return RP_new, Rpst_new, RH_new # It is better to keep the return types consistent, even though it might return None

def assert_RP_returns(RP, RP_new, Rpst, Rpst_new, RH, RH_new):
    if RP_new is not None: assert RP.shape == RP_new.shape, f"The shape of the new RP vector must be the same as the old RP vector, but it was {RP_new}"
    if Rpst_new is not None: assert Rpst.shape == Rpst_new.shape, f"The shape of the new Rpst vector must be the same as the old Rpst vector, but it was {Rpst_new}"
    if RH_new is not None: assert RH.shape == RH_new.shape, f"The shape of the new RH vector must be the same as the old RH vector, but it was {RH_new}"


def noise_decay(config_data, epoch):
    if epoch % config_data['decay_step'] == 0 and epoch != 0:
        delta = config_data['delta']
        delta *= config_data['decay'] # Decay is multiplicative. 
        config_data['delta'] = delta

def permute_indices(P_permute_idxs, pst_permute_idxs, H_permute_idxs, permute_how_many):
    total_permute = len(P_permute_idxs) + len(pst_permute_idxs) + len(H_permute_idxs)
    if permute_how_many != -1 and permute_how_many and not (permute_how_many >= total_permute):
        # If we are not permuting all, we need to choose which indices to permute.
        num_P_idxs = np.random.hypergeometric(len(P_permute_idxs), len(pst_permute_idxs) + len(H_permute_idxs), permute_how_many)
        if (permute_how_many - num_P_idxs) > 0:
            num_pst_idxs = np.random.hypergeometric(len(pst_permute_idxs), len(H_permute_idxs), permute_how_many - num_P_idxs)
            num_H_idxs = permute_how_many - num_P_idxs - num_pst_idxs
        else:
            num_pst_idxs = 0
            num_H_idxs = 0
        new_P_permute_idxs = np.random.choice(P_permute_idxs, size=num_P_idxs, replace=False)
        new_pst_permute_idxs = np.random.choice(pst_permute_idxs, size=num_pst_idxs, replace=False)
        new_H_permute_idxs = np.random.choice(H_permute_idxs, size=num_H_idxs, replace=False)
        return new_P_permute_idxs, new_pst_permute_idxs, new_H_permute_idxs
    else:
        return P_permute_idxs.astype(int), pst_permute_idxs.astype(int), H_permute_idxs.astype(int)

def permute_indices2(P_permute_idxs, pst_permute_idxs, H_permute_idxs, permute_how_many):
        chosen_idx = np.random.choice(np.arange(len(P_permute_idxs) + len(pst_permute_idxs) + len(H_permute_idxs)),
                            size=permute_how_many, replace=False)
        new_P_permute_idxs = chosen_idx[chosen_idx < len(P_permute_idxs)]
        new_pst_permute_idxs = chosen_idx[(chosen_idx >= len(P_permute_idxs)) & (chosen_idx < len(P_permute_idxs) + len(pst_permute_idxs))]
        new_H_permute_idxs = chosen_idx[chosen_idx >= len(P_permute_idxs) + len(pst_permute_idxs)]
        return new_P_permute_idxs, new_pst_permute_idxs, new_H_permute_idxs

# Thankfully this function is no longer necessary as the package is pip installable. 
def vscode_fix():
    if 'TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode':
        print("Running in VS Code, fixing sys path")
        import sys

        sys.path.append("./")

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


def get_board_arrays(game_moves):
    board = chess.Board()
    positions = []

    for move in game_moves:
        board.push_san(move)
        positions.append(board_to_array(board))

    return positions


def board_to_array(board, material_dict=None, dtype=np.int8):
    if material_dict is None:
        material_dict = {i: i for i in range(1, 7)}
    arr = np.zeros(64, dtype=dtype)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            arr[square] = material_dict[piece.piece_type] * (1 if piece.color else -1)
    # arr = arr.reshape((8, 8))     # Need a reason to reshape
    return arr


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
                        boards.append(board2sunfish(board, eval_pos(board)))
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









# def policy_walk(R, boards, moves, delta=1e-3, epochs=10, depth=3, alpha=2e-2, permute_end_idx=-1, permute_all=True,
#                 save_every=None, save_path=None, san=False, quiesce=False, n_threads=-2, plot_every=None):
#     """ Policy walk algorithm over given class of reward functions.
#     Iterates over the initial reward function by perterbing each dimension uniformly and then
#     accepting the new reward function with probability proportional to how much better they explain the given trajectories. 

#     Args:
#         R (_type_): The reward function (heuristic for statically evaluation a board).
#         boards (_type_): list of chess.Move() objects
#         moves (_type_): _description_
#         delta (_type_, optional): _description_. Defaults to 1e-3.
#         epochs (int, optional): _description_. Defaults to 10.
#         depth (int, optional): _description_. Defaults to 3.
#         alpha (_type_, optional): _description_. Defaults to 2e-2.

#     Returns:
#         _type_: _description_
#     """
#     for epoch in tqdm(range(epochs), desc='Iterating over epochs'):
#         i = 0
#         Q_newR_O = np.zeros(len(boards))
#         Q_newR_DO = np.zeros(len(boards))
#         Q_boards_oldR_DO_list = []
#         energy_newR_DO, energy_oldR_DO = 0, 0

#         R_ = copy(R)
#         if permute_all:
#             add = np.random.uniform(low=-delta, high=delta, size=R.shape[0] - 1).astype(R.dtype)
#             R_[1:permute_end_idx] += add
#         else:
#             choice = np.random.choice(np.arange(1, len(R_) if permute_end_idx < 0 else permute_end_idx))
#             R_[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

#         for board, move in tqdm(zip(boards, moves), total=len(boards), desc='Policy walking over reward functions'):
#             board.push_san(move) if san else board.push(move)
#             reward_sign = 1 if board.turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation. 
#             # First we get the board from the state/action pair seen in the data using the old weights
#             if len(Q_boards_oldR_DO_list):
#                 Q_oldR_DO_policy, board_old = Q_boards_oldR_DO_list[i]
#             else:
#                 Q_oldR_DO_policy, board_old, _ = alpha_beta_search(board=board, R=R, depth=depth-1, maximize=board.turn, quiesce=quiesce)
#                 Q_oldR_DO_policy *= reward_sign
#             # Then we evaluate the found board using the new weights
#             Q_newR_DO_policy = evaluate_board(board=board_old, RP=R_)*reward_sign     # Q^pi(s,a,R_)
#             board.pop()
#             # Finally we calculate the Q-value of the old policy on the state without the original move
#             _, board_old_, _ = alpha_beta_search(board=board, R=R, depth=depth, maximize=board.turn, quiesce=quiesce)
#             Q_newR_O_policy = evaluate_board(board=board_old_, RP=R_)*reward_sign    # Q^pi(s,pi(s),R_)

#             Q_newR_O[i] = Q_newR_O_policy     # Q^pi(s,pi(s),R_)
#             Q_newR_DO[i] = Q_newR_DO_policy   # Q^pi(s,a,R_)

#             energy_oldR_DO += Q_oldR_DO_policy
#             energy_newR_DO += Q_newR_DO_policy
#             i += 1

#         if np.sum(Q_newR_DO < Q_newR_O):
#             energy_newR_DN = 0
#             for board, move in tqdm(zip(boards, moves), total=len(boards), desc='Calculating Q-values for new Policy'):
#                 Q_newR_DN_policy, board_newR_DN, _ = alpha_beta_search(board=board, R=R_, depth=depth, maximize=board.turn, quiesce=quiesce)
#                 Q_boards_oldR_DO_list.append((Q_newR_DN_policy, board_newR_DN))
#                 energy_newR_DN += Q_newR_DN_policy
#             log_prob = min(0, log_prob_dist(R_, energy_newR_DN, alpha=alpha) - log_prob_dist(R, energy_oldR_DO, alpha=alpha))
#         else:
#             log_prob = min(0, log_prob_dist(R_, energy_newR_DO, alpha=alpha) - log_prob_dist(R, energy_oldR_DO, alpha=alpha))

#         p = np.random.rand(1).item()
#         if log_prob > -1e7 and p < np.exp(log_prob):
#             R = R_
#         if save_every is not None and epoch % save_every == 0:
#             pd.DataFrame(R_.reshape((-1, 1)), columns=['Result']).to_csv(join(save_path, f'{epoch}.csv'), index=False)
#         if plot_every is not None and epoch % plot_every == 0:
#             plot_R_weights(epochs=epochs, save_every=save_every, out_path=save_path, )

#     return R


# def policy_walk_multi(R, boards, moves, delta=1e-3, epochs=10, depth=3, alpha=2e-2, permute_end_idx=-1, permute_all=True,
#                       save_every=None, save_path=None, san=False, quiesce=False, n_threads=-2, plot_every=None):
#     """ Policy walk algorithm over given class of reward functions.
#     Iterates over the initial reward function by perterbing each dimension uniformly and then
#     accepting the new reward function with probability proportional to how much better they explain the given trajectories.

#     Args:
#         R (_type_): The reward function (heuristic for statically evaluation a board).
#         boards (_type_): list of chess.Move() objects
#         moves (_type_): _description_
#         delta (_type_, optional): _description_. Defaults to 1e-3.
#         epochs (int, optional): _description_. Defaults to 10.
#         depth (int, optional): _description_. Defaults to 3.
#         alpha (_type_, optional): _description_. Defaults to 2e-2.

#     Returns:
#         _type_: _description_
#     """

#     # Multiprocessesing
#     def step(board, move, Q_boards_oldR_DO_list, R, depth):
#         reward_sign = 1 if board.turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation. 
#         board.push_san(move) if san else board.push(move)
#         # First we get the board from the state/action pair seen in the data using the old weights
#         if len(Q_boards_oldR_DO_list):
#             Q_oldR_DO_policy, board_old = Q_boards_oldR_DO_list[i]
#         else:
#             Q_oldR_DO_policy, board_old, _ = alpha_beta_search(board=board, R=R, depth=depth - 1, maximize=board.turn, quiesce=quiesce)
#             Q_oldR_DO_policy *= reward_sign
#         # Then we evaluate the found board using the new weights
#         Q_newR_DO_policy = evaluate_board(board=board_old, RP=R_, white=board_old.turn)*reward_sign  # Q^pi(s,a,R_)
#         board.pop()
#         # Finally we calculate the Q-value of the old policy on the state without the original move
#         _, board_old_, _ = alpha_beta_search(board=board, R=R, depth=depth, maximize=board.turn, quiesce=quiesce)
#         Q_newR_O_policy = evaluate_board(board=board, RP=R_, white=board_old_.turn)*reward_sign
#         return Q_newR_O_policy, Q_newR_DO_policy, Q_oldR_DO_policy

#     for epoch in tqdm(range(epochs), desc='Iterating over epochs'):
#         i = 0
#         Q_newR_O = np.zeros(len(boards))
#         Q_newR_DO = np.zeros(len(boards))
#         Q_boards_oldR_DO_list = []
#         energy_newR_DO, energy_oldR_DO = 0, 0

#         R_ = R
#         if permute_all:
#             add = np.random.uniform(low=-delta, high=delta, size=R.shape[0] - 1).astype(R.dtype)
#             R_[1:permute_end_idx] += add
#         else:
#             choice = np.random.choice(np.arange(1, len(R_) if permute_end_idx < 0 else permute_end_idx))
#             R_[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

#         result = Parallel(n_jobs=n_threads)(delayed(step)(board, moves, Q_boards_oldR_DO_list, R, depth)
#                                             for board, moves in tqdm(zip(boards, moves), total=len(boards),
#                                                               desc='Policy walking over reward functions'))

#         for Q_newR_O_policy, Q_newR_DO_policy, Q_oldR_DO_policy in result:
#             Q_newR_O[i] = Q_newR_O_policy  # Q^pi(s,pi(s),R_)
#             Q_newR_DO[i] = Q_newR_DO_policy  # Q^pi(s,a,R_)

#             energy_oldR_DO += Q_oldR_DO_policy
#             energy_newR_DO += Q_newR_DO_policy
#             i += 1

#         if np.sum(Q_newR_DO < Q_newR_O):
#             energy_newR_DN = 0
#             result = Parallel(n_jobs=-2)(delayed(alpha_beta_search)(board=board, R=R_, depth=depth,
#                                                                     maximize=board.turn, quiesce=quiesce)
#                                          for board, move in tqdm(zip(boards, moves), total=len(boards),
#                                                                  desc='Calculating Q-values for new Policy'))
#             for Q_newR_DN_policy, board_newR_DN, _ in result:
#                 Q_boards_oldR_DO_list.append((Q_newR_DN_policy, board_newR_DN))
#                 energy_newR_DN += Q_newR_DN_policy
#             log_prob = min(0, log_prob_dist(R_, energy_newR_DN, alpha=alpha) - log_prob_dist(R, energy_oldR_DO,
#                                                                                              alpha=alpha))
#         else:
#             log_prob = min(0, log_prob_dist(R_, energy_newR_DO, alpha=alpha) - log_prob_dist(R, energy_oldR_DO,
#                                                                                              alpha=alpha))

#         p = np.random.rand(1).item()
#         if log_prob > -1e7 and p < np.exp(log_prob):
#             R = R_
#         if save_every is not None and epoch % save_every == 0:
#             pd.DataFrame(R_.reshape((-1, 1)), columns=['Result']).to_csv(join(save_path, f'{epoch}.csv'), index=False)
#         if plot_every is not None and epoch % plot_every == 0:
#             plot_R_weights(epochs=epochs, save_every=save_every, out_path=save_path, )

#     return R

# def policy_walk_v0_multi(R, boards, moves, config_data, out_path):
#     """ Policy walk algorithm over given class of reward functions.
#     Iterates over the initial reward function by perterbing each dimension uniformly and then
#     accepting the new reward function with probability proportional to how much better they explain the given trajectories.

#     Args:
#         R (_type_): The reward function (heuristic for statically evaluation a board).
#         boards (_type_): list of chess.Move() objects
#         moves (_type_): _description_
#         delta (_type_, optional): _description_. Defaults to 1e-3.
#         epochs (int, optional): _description_. Defaults to 10.
#         depth (int, optional): _description_. Defaults to 3.
#         alpha (_type_, optional): _description_. Defaults to 2e-2.

#     Returns:
#         _type_: _description_
#     """
#     delta = config_data['delta']
#     depth = config_data['search_depth']
#     k = config_data['search_depth']
#     epochs = config_data['epochs']
#     save_every = config_data['save_every']
#     permute_all = config_data['permute_all']
#     permute_idxs = char_to_idxs(config_data['permute_char'])
#     quiesce = config_data['quiesce']
#     n_threads = config_data['n_threads']
#     plot_every = config_data['plot_every']


#     # Multiprocessesing
#     def step(board, move, R, R_, depth, k):
#         reward_sign = 1 if board.turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
#         # Finally we calculate the Q-value of the old policy on the state without the original move
#         k_best_moves_old = alpha_beta_search_k(board=board, RP=R, depth=depth, k=k, maximize=board.turn, quiesce=quiesce, Rpst=True)
#         k_best_moves_new = alpha_beta_search_k(board=board, RP=R_, depth=depth, k=k, maximize=board.turn, quiesce=quiesce, Rpst=True)

#         return move in list_first_moves(k_best_moves_old), move in list_first_moves(k_best_moves_new)

#     for epoch in tqdm(range(epochs), desc='Iterating over epochs'):
#         i = 0

#         R_ = copy(R)
#         if permute_all:
#             add = np.random.uniform(low=-delta, high=delta, size=R.shape[0] - 1).astype(R.dtype)
#             R_[permute_idxs] += add
#         else:
#             choice = np.random.choice(permute_idxs)
#             R_[choice] += np.random.uniform(low=-delta, high=delta, size=1).item()

#         result = Parallel(n_jobs=n_threads)(delayed(step)(board, moves, R, R_, depth, k)
#                                             for board, moves in tqdm(zip(boards, moves), total=len(boards),
#                                                               desc='Policy walking over reward functions'))

#         result_array = np.array(result)

#         if np.argmax(result_array.sum(axis=0)):
#             R = copy(R_)
#         if save_every is not None and epoch % save_every == 0:
#             pd.DataFrame(R_.reshape((-1, 1)), columns=['Result']).to_csv(join(out_path, f'{epoch}.csv'), index=False)
#         if plot_every is not None and epoch % plot_every == 0:
#             plot_R_weights(config_data=config_data, out_path=out_path, )

#     return R


# def policy_walk_depth(R, boards, moves, delta=1e-3, epochs=10, depth_max=3, alpha=2e-2, time_max=np.inf,
#                       timer_moves=False):
#     """ Policy walk algorithm over the depth of the search.
#     Begins with a uniform distribution over search depths and iterates by perterbing each dimension uniformly and then
#     accepting the new softmax distribution of search depths with probability proportional to how much better they explain the given trajectories. 

#     Args:
#         R (_type_): The reward function (heuristic for statically evaluation a board).
#         boards (_type_): _description_
#         moves (_type_): _description_
#         delta (_type_, optional): _description_. Defaults to 1e-3.
#         epochs (int, optional): _description_. Defaults to 10.
#         depth_max (int, optional): _description_. Defaults to 3.
#         alpha (_type_, optional): _description_. Defaults to 2e-2.
#         time_max (_type_, optional): _description_. Defaults to np.inf.
#         timer_moves (bool, optional): _description_. Defaults to False.

#     Returns:
#         ndarray : final array of search depth probability distribution pre-softmax. Indexes signify depth starting from 1. 
#     """
#     print("policy_walk_depth, to be fixed")
#     depth_dist = np.ones(depth_max)
#     start = time()
#     for epoch in tqdm(range(epochs)):
#         Q_moves = np.zeros(len(boards))
#         Q_policy = np.zeros(len(boards))
#         i = 0
#         energy_new, energy_old = 0, 0
#         for board, move in tqdm(zip(boards, moves), total=len(boards)):
#             add = np.random.uniform(low=-delta, high=delta, size=depth_dist.shape[0]).astype(
#                 depth_dist.dtype)  # * (delta / 2)
#             depth_dist_ = depth_dist + add

#             board.push(move)
#             depth1 = softmax_choice(depth_dist)
#             depth2 = softmax_choice(depth_dist_)
#             _, Q_old = get_best_move(board=board, R=R, depth=depth1, timer=timer_moves, white=board.turn)
#             _, Q_new = get_best_move(board=board, R=R, depth=depth2, timer=timer_moves, white=board.turn)
#             if Q_new is None or Q_old is None:
#                 continue
#             board.pop()

#             Q_moves[i] = Q_old
#             Q_policy[i] = Q_new

#             energy_old += Q_old
#             energy_new += Q_new

#             i += 1
#             # prob = min(1, prob_dist(depth_dist_, energy_new, alpha=alpha)/prob_dist(depth_dist, energy_old, alpha=alpha))

#             log_prob = min(0,
#                            log_prob_dist(depth_dist_, energy_new, alpha=alpha) - log_prob_dist(depth_dist, energy_old,
#                                                                                                alpha=alpha))

#             if np.sum(Q_policy < Q_moves):
#                 if log_prob > -1e7 and np.random.rand(1).item() < np.exp(log_prob):
#                     depth_dist = copy(depth_dist_)
#             if start - time() > time_max:
#                 return depth_dist
#     return depth_dist
