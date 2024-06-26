import chess
import numpy as np
from tqdm import tqdm
import logging
from copy import deepcopy
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos, \
    get_new_pst, str_to_sunfish_move, sunfish_move_to_str, moves_and_Q_from_result, sunfish_move
from irl_chess.chess_utils.sunfish import piece, pst, pst_only, Position, Move
from irl_chess.visualizations.visualize import plot_R_weights, char_to_idxs
from irl_chess.chess_utils.alpha_beta_utils import no_moves_eval, evaluate_board, alpha_beta_search, alpha_beta_search_k, list_first_moves 
from irl_chess.chess_utils.utils import perturb_reward
# from typing import List
from joblib import Parallel, delayed

# ======================== Functions for BIRL policy walk ====================== #
def log_prob_dist(R, energy, alpha, prior=lambda R: 1):
    log_prob = alpha * energy + np.log(prior(R))
    return log_prob

def eval_log_prob(RP_new, QpiNew, RP, Qpi, alpha):
    log_prob = min(0, log_prob_dist(RP_new, np.sum(QpiNew), alpha=alpha) - log_prob_dist(RP, np.sum(Qpi), alpha=alpha))
    return log_prob

def PolicyIteration(states, actions, RP, Rpst, RH, config_data, parallel = None):
    pass

def pi_alpha_beta_search(states, actions, RP, Rpst, RH, config_data, parallel = None):
    pi = [None] * len(states)
    Qpi_policy_R = np.zeros(len(states))
    pi_moves = [None] * len(states)
    for i, s in enumerate(states):
        # Interpret following policy pi as alpha_beta searching with evaluation function R. board_final is interpreted as the culmination of the Q-value-iteration,
        # and its evaluation is therefore a reasonable approximation of the Q-value, as we at least get an evaluation of that if our opponent plays optimally according 
        # to our reward function (all given the approximations inherent in limited search depth).
        assert isinstance(s, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s)}"
        reward_sign = 1 if s.turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        if actions is not None: # Follow policy pi after taking move a
            a = actions[i]
            if isinstance(a, chess.Move):
                s.push(a)
            elif isinstance(a, str):
                s.push_san(a)  # s.push_san(move)
            else:
                raise TypeError
            if s.outcome() is not None:
                eval, board_final, _ = no_moves_eval(s, RP, Rpst, RH)
                # board_final = deepcopy(s)
                s.pop()
                return eval * reward_sign, board_final, None
        eval, board_final, move_trajectory = alpha_beta_search(board=s, RP=RP, depth=config_data['depth'] - (1 if actions is not None else 0), 
                                                 maximize=s.turn, Rpst = Rpst, RH=RH, quiesce=config_data['quiesce'])
        if actions is not None: s.pop()
        Qpi_policy_R[i], pi[i], pi_moves[i] = eval*reward_sign, board_final, move_trajectory[0]
    return pi, Qpi_policy_R, pi_moves

def pi_alpha_beta_search_par(states, actions, RP, Rpst, RH, config_data, parallel):
    def evaluate_single_state(i, s):
        # Interpret following policy pi as alpha_beta searching with evaluation function R.
        assert isinstance(s, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s)}"
        reward_sign = 1 if s.turn else -1  # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        if actions is not None: # Follow policy pi after taking move a
            a = actions[i]
            if isinstance(a, chess.Move):
                s.push(a)
            elif isinstance(a, str):
                s.push_san(a)  # s.push_san(move)
            else:
                raise TypeError
            if s.outcome() is not None:
                eval, board_final, _ = no_moves_eval(s, RP, Rpst, RH)
                # board_final = deepcopy(s)
                s.pop()
                return eval * reward_sign, board_final, None
        eval, board_final, move_trajectory = alpha_beta_search(
            board=s, RP=RP, depth=config_data['depth'] - (1 if actions is not None else 0),
            maximize=s.turn, Rpst = Rpst, RH=RH, quiesce=config_data['quiesce']
        )
        if actions is not None: s.pop()
        return eval * reward_sign, board_final, move_trajectory[0]
    
    results = parallel(delayed(evaluate_single_state)(i, s) for i, s in enumerate(states))
    pi, Qpi_policy_R, pi_moves = zip(*[(board_final, eval, first_move_uci) for eval, board_final, first_move_uci in results])
    Qpi_policy_R = np.array(Qpi_policy_R)
    return list(pi), Qpi_policy_R, list(pi_moves)

def sunfish_search(states, actions, RP, Rpst, RH, config_data, parallel):
    pi = [None] * len(states)
    Qpi_policy_R = np.zeros(len(states))
    pi_moves = [None] * len(states)
    pst = get_new_pst(RP, Rpst)
    for i, s in enumerate(states):
        assert isinstance(s, Position), f"For sunfish policy, states must be of type Position, but got {type(s)}"
        # Sunfish always seeks to maximize the score and views each position as white.
        if actions is not None: # Follow policy pi after taking move a
            a = actions[i]
            assert isinstance(a, Move)
            s_a = s.move(Move, pst)     # Searches will end at different final depths, but that is not a problem as both are following the policy
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s_a, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            if best_move is not None:
                eval = -1 * best_board_found_tuple[1]  # Invert because s_a was from the perspective of the opposite player
            else: # If there are no legal moves, next state is temrinal and the evaluation is the reward gained in making that move.
                eval = (-1 * eval_pos(s_a, RP, Rpst) - eval_pos(s, RP, Rpst))
        else:
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = best_board_found_tuple[1]

        board_final_opposite_player = (best_board_found_tuple[0], best_board_found_tuple[2])
        Qpi_policy_R[i], pi[i], pi_moves[i] = eval, board_final_opposite_player, best_move
    return pi, Qpi_policy_R, pi_moves


def sunfish_search_par(states, actions, RP, Rpst, RH, config_data, parallel):
    def evaluate_single_state(i, s, pst):
        logging.debug(f"Evaluating state {i}")
        assert isinstance(s, Position), f"For sunfish policy, states must be of type Position, but got {type(s)}"
        # Sunfish always seeks to maximize the score and views each position as white.
        if actions is not None:
            # Follow policy pi after taking move a
            a = actions[i]
            logging.debug(f"Taking action {a}")
            assert isinstance(a, Move)
            s_a = s.move(a, pst)  # Searches will end at different final depths, but that is not a problem as both are following the policy
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s_a, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            if best_move is not None:
                eval = -1 * best_board_found_tuple[1]  # Invert because s_a was from the perspective of the opposite player
            else: # If there are no legal moves, next state is temrinal and the evaluation is the reward gained in making that move.
                logging.info(f"No best move found after action. The position was: {s.board}, which transitioned to {s_a.board}")
                eval = (-1 * eval_pos(s_a, RP, Rpst) - eval_pos(s, RP, Rpst))
                return eval, (s_a, True), None 
        else:
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = best_board_found_tuple[1]

        board_final_opposite_player = (best_board_found_tuple[0], best_board_found_tuple[2]) # Is a tuple of (board, opposite_player)
        return eval, board_final_opposite_player, best_move

    pst = get_new_pst(RP, Rpst)
    if actions is not None and len(states) != len(actions):
        raise ValueError("The length of states and actions must be the same.")
    logging.debug("Starting parallel execution")
    results = parallel(delayed(evaluate_single_state)(i, s, pst) for i, s in enumerate(states))
    logging.debug("Parallel execution completed")

    pi, Qpi_policy_R, pi_moves = zip(*[(board_final_opposite_player, eval, best_move) for eval, board_final_opposite_player, best_move in results])
    Qpi_policy_R = np.array(Qpi_policy_R)
    return list(pi), Qpi_policy_R, list(pi_moves)

def Qeval(pi, states, RP, Rpst, RH, parallel, evaluation_function = evaluate_board):
    pass

def Qeval_chessBoard(pi, states, RP, Rpst, RH, parallel = None, evaluation_function = evaluate_board):
    Qpi_R = np.zeros(len(states))
    for i, s_final in enumerate(pi):
        assert isinstance(s_final, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s_final)}"
        reward_sign = 1 if states[i].turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        Qpi_R[i] = no_moves_eval(s_final, RP, Rpst, RH=RH, evaluation_function=evaluation_function)[0] * reward_sign
    return Qpi_R

def Qeval_chessBoard_par(pi, states, RP, Rpst, RH, parallel, evaluation_function = evaluate_board):
    def evaluate_single_board(i, s_final):
        assert isinstance(s_final, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s_final)}"
        reward_sign = 1 if states[i].turn else -1  # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        return no_moves_eval(s_final, RP, Rpst, RH=RH, evaluation_function=evaluation_function)[0] * reward_sign
    Qpi_R = parallel(delayed(evaluate_single_board)(i, s_final) for i, s_final in enumerate(pi))
    return np.array(Qpi_R)

def Qeval_sunfishBoard_par(pi, states, RP, Rpst, RH, parallel, evaluation_function = evaluate_board):
    def evaluate_single_board(i, s_f_tuple):
        s_final, opposite_player = s_f_tuple
        assert isinstance(s_final, Position), f"For sunfish policy, states must be of type Position, but got {type(s_final)}"
        reward_sign = -1 if opposite_player else 1 # Is s_final from the perspective of the opponent or not.
        return  eval_pos(s_final, RP, Rpst) * reward_sign
    Qpi_R = parallel(delayed(evaluate_single_board)(i, s_f_tuple) for i, s_f_tuple in enumerate(pi))
    return np.array(Qpi_R)

def bookkeeping(accuracies, actions, pi_moves, pi_energies, a_energies, Qpi_policy_R, Qpi_action_R, RPs, RP, Rpsts= None, Rpst = None, RHs = None, RH = None):
    acc = sum([player_move == policy_move for player_move, policy_move in list(zip(actions, pi_moves))]) / len(actions)
    accuracies.append(acc)
    pi_energies.append(np.sum(Qpi_policy_R))
    a_energies.append(np.sum(Qpi_action_R))
    RPs.append(RP)
    if Rpsts is not None:
        Rpsts.append(Rpst)
    if RHs is not None:
        RHs.append(RH)
    return acc

# Not used anywhere, but was the template for run_BIRL
# Actual function. Criteria: variable move source. Variable searcher. Variable switching criteria. Variable board type. 
def base_policy_walk(states, actions, R, config_data, PolicyIteration, Qeval):

    # Define variables. If a veriable is modified, it is passed as input and overwritten by being assigned to the output, with the exception of bookkeeping().
    # Interpret following the policy as arriving at the same final board. 
    pi, a_pi, pi_new = [None] * len(states), [None] * len(states), [None] * len(states) 
    Qpi_policy_R    = np.zeros(len(states)) # Qpi(s,pi,R)
    Qpi_action_Rnew = np.zeros(len(states)) # Qpi(s,a ,R~)
    Qpi_policy_Rnew = np.zeros(len(states)) # Qpi(s,pi,R~)
    QpiNew_policy_Rnew = np.zeros(len(states)) # Qpi~(s,pi~,R~)
    accuracies = []
    energies = []

    # Calculate initially. 
    pi, Qpi_policy_R, pi_moves = PolicyIteration(states, None, R, config_data)
    a_pi, _, _ = PolicyIteration(states, actions, R, config_data) # We can't guarantee that alpha-beta search fully explores move a and so we calculate it again.
    bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R)
    

    for epoch in tqdm(range(0, config_data['epochs']), desc='Iterating over epochs'):
        print(f'Epoch {epoch}\n', '-' * 25)

        R_new = perturb_reward(R, config_data['permute_idxs'], config_data['delta'], config_data['noise_distribution'], config_data['permute_how_many'])
        
        # Evaluate perterbued reward function
        Qpi_action_Rnew = Qeval(a_pi, states, R_new)
        Qpi_policy_Rnew = Qeval(pi, states, R_new) # This the standard Q-value there, the policy should be optimal. 

        # Switch stochastically accept the new reward function and the new policy
        if np.any(Qpi_policy_Rnew < Qpi_action_Rnew): # if the new reward function explains the data action better than the policy action for any state
            pi_new, QpiNew_policy_Rnew, pi_new_moves = PolicyIteration(states, None, R_new, config_data)
            log_prob = min(0, log_prob_dist(R_new, np.sum(QpiNew_policy_Rnew), alpha=config_data['alpha']) - log_prob_dist(R, np.sum(Qpi_policy_R), alpha=config_data['alpha']))

            if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                print(f'Changed weights and policy! From {R}\n to {R_new}\n Probability was: {np.exp(log_prob)}')
                R = np.copy(R_new)
                pi, Qpi_policy_R, pi_moves = pi_new.copy(), np.copy(QpiNew_policy_Rnew), pi_new_moves.copy()
                a_pi, _, _ = PolicyIteration(states, a_pi, actions, Qpi_policy_R, R, config_data)
        else:
            log_prob = min(0, log_prob_dist(R_new, np.sum(Qpi_policy_Rnew), alpha=config_data['alpha']) - log_prob_dist(R, np.sum(Qpi_policy_R), alpha=config_data['alpha']))
            if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                print(f'Changed weights! From {R}\n to {R_new}\n Probability was: {np.exp(log_prob)}')
                Qpi_policy_R = np.copy(Qpi_policy_Rnew)
                R = np.copy(R_new)

        bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R)



# Pseudocode. 
# def policy_walk():
#     R = set reward
#     pi = policy

#     for epoch in range(0, config_data['epochs']):
#         print(f'Epoch {epoch}\n', '-' * 25)
#         R_new = perturb_reward()

#         Qpi_action_Rnew = 
#         Qpi_policy_Rnew = 

#         if exists:
#             pi_new = 
#             QpiNew_policyNew_Rnew = 
#             if prob:
#                 R = R_new
#                 pi = pi_new
#         else:
#             if prob:
#                 R = R_new

def set_chess_policy(chess_boards, player_moves, config_data):
    if config_data['chess_policy'] == "alpha_beta":
        from irl_chess.chess_utils.BIRL_utils import pi_alpha_beta_search_par as PolicyIteration, \
                Qeval_chessBoard_par as Qeval
        states = chess_boards
        actions = player_moves
    elif config_data['chess_policy'] == "sunfish":
        from irl_chess.chess_utils.BIRL_utils import sunfish_search_par as PolicyIteration, \
                Qeval_sunfishBoard_par as Qeval
        states = [board2sunfish(board, 0) for board in chess_boards] # sunfish scores are relative, so setting them to 0 is fine. 
        actions = [str_to_sunfish_move(move, not board.turn) for move, board in zip(player_moves, chess_boards)]
    else:
        raise Exception(f"The policy {config_data['chess_policy']} is not implemented yet")
    return states, actions, PolicyIteration, Qeval

def probability_of_switching(states, actions, pi, a_pi, RP_new, RP, Rpst_new, Rpst, RH_new, RH, Qpi_policy_R, Qpi_action_R, alpha, config_data, Qeval, PolicyIteration, parallel):
    Qpi_action_Rnew = Qeval(a_pi, states, RP_new, Rpst_new, RH_new, parallel)
    Qpi_policy_Rnew = Qeval(pi, states, RP_new, Rpst_new, RH_new, parallel)
    if np.any(Qpi_policy_Rnew < Qpi_action_Rnew): # if the new reward function explains the data action better than the policy action for any state
        print("There exists a state-action pair (s, a) such that Qpi(s, pi, R~) < Qpi(s, a, R~)")
        pi_new, QpiNew_policy_Rnew, pi_new_moves = PolicyIteration(states, None, RP_new, Rpst_new, RH_new, config_data, parallel)
        a_pi_new, QpiNew_action_Rnew, _ = PolicyIteration(states, actions, RP_new, Rpst_new, RH_new, config_data, parallel)
        if config_data['energy_optimized'] == "policy":
            log_prob = eval_log_prob(RP_new, QpiNew_policy_Rnew, RP, Qpi_policy_R, config_data['alpha'])
            print(f"Probability of switching R and policy is: {np.exp(log_prob)}")
            print(f"Relative energies are {np.sum(QpiNew_policy_Rnew)} to {np.sum(Qpi_policy_R)}")
        elif config_data['energy_optimized'] == "action":
            log_prob = eval_log_prob(RP_new, QpiNew_action_Rnew, RP, Qpi_action_R, config_data['alpha'])
            print(f"Probability of switching R and policy is: {np.exp(log_prob)}")
            print(f"Relative energies are {np.sum(QpiNew_action_Rnew)} to {np.sum(Qpi_action_R)}")

    else:
        print("Action is worse or equal to policy in all states under R~")
        if config_data['energy_optimized'] == "policy":
            log_prob = eval_log_prob(RP_new, Qpi_policy_Rnew, RP, Qpi_policy_R, config_data['alpha'])
            print(f"Probability of switching R is: {np.exp(log_prob)}")
            print(f"Relative energies are {np.sum(Qpi_policy_Rnew)} to {np.sum(Qpi_policy_R)}")
        elif config_data['energy_optimized'] == "action":
            log_prob = eval_log_prob(RP_new, Qpi_action_Rnew, RP, Qpi_action_R, config_data['alpha'])
            print(f"Probability of switching R is: {np.exp(log_prob)}")
            print(f"Relative energies are {np.sum(Qpi_action_Rnew)} to {np.sum(Qpi_action_R)}")
    return np.exp(log_prob)