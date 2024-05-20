from joblib import Parallel, delayed
import chess
import numpy as np
from tqdm import tqdm
from irl_chess.chess_utils.sunfish_utils import board2sunfish, eval_pos
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move, sunfish_move_to_str, moves_and_Q_from_result
from irl_chess.chess_utils.sunfish import piece, pst, pst_only, Position, Move
from irl_chess.visualizations.visualize import plot_R_weights, char_to_idxs
from irl_chess.chess_utils.alpha_beta_utils import no_moves_eval, evaluate_board, alpha_beta_search, alpha_beta_search_k, list_first_moves 
from scipy.special import softmax
from irl_chess.chess_utils.utils import perturb_reward
from typing import List
from joblib import Parallel, delayed
from irl_chess.models.sunfish_GRW import eval_pos, sunfish_move

# ======================== Functions for BIRL policy walk ====================== #
def log_prob_dist(R, energy, alpha, prior=lambda R: 1):
    log_prob = alpha * energy + np.log(prior(R))
    return log_prob
    
def PolicyIteration(states, pi, actions, Qpi_policy_R, R, config_data):
    pass

def pi_alpha_beta_search(states, pi, actions, Qpi_policy_R, R, config_data, parallel):
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
        eval, board_final, move_trajectory = alpha_beta_search(board=s, R=R, depth=config_data['depth'] - (1 if actions is not None else 0), 
                                                 maximize=s.turn, pst = config_data['pst'], quiesce=config_data['quiesce'])
        if actions is not None: s.pop()
        Qpi_policy_R[i], pi[i], pi_moves[i] = eval*reward_sign, board_final, move_trajectory[0]
    return pi, Qpi_policy_R, pi_moves

def pi_alpha_beta_search_par(states, pi, actions, Qpi_policy_R, R, config_data, parallel):
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
        eval, board_final, move_trajectory = alpha_beta_search(
            board=s, R=R, depth=config_data['depth'] - (1 if actions is not None else 0),
            maximize=s.turn, pst = config_data['pst'], quiesce=config_data['quiesce']
        )
        if actions is not None: s.pop()
        return eval * reward_sign, board_final, move_trajectory[0]
    
    pi_moves = [None] * len(states)
    results = parallel(delayed(evaluate_single_state)(i, s) for i, s in enumerate(states))
    for i, (eval, board_final, first_move_uci) in enumerate(results):
        Qpi_policy_R[i] = eval
        pi[i] = board_final
        pi_moves[i] = first_move_uci

    return pi, Qpi_policy_R, pi_moves

def sunfish_search(states, pi, actions, Qpi_policy_R, R, config_data, parallel):
    pi_moves = [None] * len(states)
    pst = get_new_pst(R)
    for i, s in enumerate(states):
        assert isinstance(s, Position), f"For sunfish policy, states must be of type Position, but got {type(s)}"
        # Sunfish always seeks to maximize the score and views each position as white.
        if actions is not None: # Follow policy pi after taking move a
            a = actions[i]
            assert isinstance(a, Move)
            s_a = s.move(Move, pst)     # Searches will end at different final depths, but that is not a problem as both are following the policy
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s_a, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = -1*best_board_found_tuple[1] # Invert because s_a was from the perspective of the opposite player
        else:
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = best_board_found_tuple[1]

        board_final_opposite_player = (best_board_found_tuple[0], best_board_found_tuple[2])
        Qpi_policy_R[i], pi[i], pi_moves[i] = eval, board_final_opposite_player, best_move
    return pi, Qpi_policy_R, pi_moves

def sunfish_search_par(states, pi, actions, Qpi_policy_R, R, config_data, parallel):
    def evaluate_single_state(i, s):
        assert isinstance(s, Position), f"For sunfish policy, states must be of type Position, but got {type(s)}"
        # Sunfish always seeks to maximize the score and views each position as white.
        if actions is not None:
            # Follow policy pi after taking move a
            a = actions[i]
            assert isinstance(a, Move)
            s_a = s.move(a, pst)  # Searches will end at different final depths, but that is not a problem as both are following the policy
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s_a, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = -1 * best_board_found_tuple[1]  # Invert because s_a was from the perspective of the opposite player
        else:
            best_move, best_moves, move_dict, best_board_found_tuple = \
                sunfish_move(s, pst, time_limit=config_data['time_limit'], min_depth=2, return_best_board_found_tuple=True)
            eval = best_board_found_tuple[1]

        board_final_opposite_player = (best_board_found_tuple[0], best_board_found_tuple[2])
        return eval, board_final_opposite_player, best_move

    pst = get_new_pst(R)
    results = parallel(delayed(evaluate_single_state)(i, s) for i, s in enumerate(states))
    
    pi_moves = [None] * len(states)
    for i, (eval, board_final_opposite_player, best_move) in enumerate(results):
        Qpi_policy_R[i] = eval
        pi[i] = board_final_opposite_player
        pi_moves[i] = best_move

    return pi, Qpi_policy_R, pi_moves

def Qeval(pi, states, R, parallel, pst = False, evaluation_function = evaluate_board):
    pass

def Qeval_chessBoard(pi, states, R, parallel = None, pst = False, evaluation_function = evaluate_board):
    Qpi_R = np.zeros(len(states))
    for i, s_final in enumerate(pi):
        assert isinstance(s_final, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s_final)}"
        reward_sign = 1 if states[i].turn else -1 # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        Qpi_R[i] = no_moves_eval(s_final, R, pst, evaluation_function)[0] * reward_sign
    return Qpi_R

def Qeval_chessBoard_par(pi, states, R, parallel, pst = False, evaluation_function = evaluate_board):
    def evaluate_single_board(i, s_final):
        assert isinstance(s_final, chess.Board), f"For alpha beta policy, states must be of type chess.Board, but got {type(s_final)}"
        reward_sign = 1 if states[i].turn else -1  # White seeks to maximize and black to minimize, so the reward for black is the flipped evaluation.
        Qpi_R[i] = no_moves_eval(s_final, R, pst, evaluation_function=evaluation_function)[0] * reward_sign
    Qpi_R = np.zeros(len(states))
    parallel(delayed(evaluate_single_board)(i, s_final) for i, s_final in enumerate(pi))
    return Qpi_R

def Qeval_sunfishBoard_par(pi, states, R, parallel, pst = False, evaluation_function = evaluate_board):
    def evaluate_single_board(i, s_f_tuple):
        s_final, opposite_player = s_f_tuple
        assert isinstance(s_final, Position), f"For sunfish policy, states must be of type Position, but got {type(s_final)}"
        reward_sign = -1 if opposite_player else 1 # Is s_final from the perspective of the opponent or not.
        Qpi_R[i] = eval_pos(s_final, R) * reward_sign
    Qpi_R = np.zeros(len(states))
    parallel(delayed(evaluate_single_board)(i, s_f_tuple) for i, s_f_tuple in enumerate(pi))
    return Qpi_R

def bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R, Rs, R):
    acc = sum([player_move == policy_move for player_move, policy_move in list(zip(actions, pi_moves))]) / len(actions)
    accuracies.append(acc)
    energies.append(np.sum(Qpi_policy_R))
    Rs.append(R)
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
    pi, Qpi_policy_R, pi_moves = PolicyIteration(states, pi, None, Qpi_policy_R, R, config_data)
    a_pi, _, _ = PolicyIteration(states, a_pi, actions, Qpi_policy_R, R, config_data) # We can't guarantee that alpha-beta search fully explores move a and so we calculate it again.
    bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R)
    

    for epoch in tqdm(range(0, config_data['epochs']), desc='Iterating over epochs'):
        print(f'Epoch {epoch}\n', '-' * 25)

        R_new = perturb_reward(R, config_data['permute_idxs'], config_data['delta'], config_data['noise_distribution'], config_data['permute_how_many'])
        
        # Evaluate perterbued reward function
        Qpi_action_Rnew = Qeval(Qpi_action_Rnew, a_pi, states, R_new)
        Qpi_policy_Rnew = Qeval(Qpi_policy_Rnew, pi, states, R_new) # This the standard Q-value there, the policy should be optimal. 

        # Switch stochastically accept the new reward function and the new policy
        if np.any(Qpi_policy_Rnew < Qpi_action_Rnew): # if the new reward function explains the data action better than the policy action for any state
            pi_new, QpiNew_policy_Rnew, pi_new_moves = PolicyIteration(states, pi_new, None, QpiNew_policy_Rnew, R_new, config_data)
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