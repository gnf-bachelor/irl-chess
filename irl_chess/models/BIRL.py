import os
from os.path import join
from time import time
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from irl_chess import pst
from irl_chess.visualizations import char_to_idxs

from irl_chess.misc_utils.utils import reformat_list
from irl_chess.misc_utils.load_save_utils import process_epoch
from irl_chess.chess_utils.sunfish_utils import board2sunfish, get_new_pst, str_to_sunfish_move, eval_pos
from irl_chess.chess_utils.BIRL_utils import pi_alpha_beta_search, pi_alpha_beta_search_par, \
    Qeval_chessBoard, Qeval_chessBoard_par, bookkeeping, perturb_reward, log_prob_dist
from irl_chess.models.sunfish_GRW import eval_pos, sunfish_move, val_util

def BIRL_result_string(model_config_data):
    chess_policy = model_config_data['chess_policy']
    delta = model_config_data['delta']
    decay = model_config_data['decay']
    decay_step = model_config_data['decay_step']
    permute_how_many = model_config_data['permute_how_many']
    R_true = reformat_list(model_config_data['R_true'], '_')
    R_start = reformat_list(model_config_data['R_start'], '_')
    if chess_policy == "alpha_beta":
        depth = model_config_data['depth']
        quisce = model_config_data["quiesce"]
        return f"{chess_policy}-{int(delta)}-{decay}-{decay_step}-{permute_how_many}-{depth}-{quisce}--{R_start}-{R_true}"
    elif chess_policy == "sunfish":
        time_limit = model_config_data['time_limit']
        return f"{chess_policy}-{int(delta)}-{decay}-{decay_step}-{permute_how_many}-{time_limit}--{R_start}-{R_true}"
    

def run_BIRL(chess_boards, player_moves, config_data, out_path, validation_set):
    if config_data['move_function'] == "sunfish_move":
        use_player_move = False
    elif config_data['move_function'] == "player_move":
        use_player_move = True
    else:
        raise Exception(f"The move function {config_data['move_function']} is not implemented yet")

    if config_data['chess_policy'] == "alpha_beta":
        from irl_chess.chess_utils.BIRL_utils import pi_alpha_beta_search_par as PolicyIteration, \
                Qeval_chessBoard_par as Qeval
        states = chess_boards
        actions = player_moves
    elif config_data['chess_policy'] == "sunfish":
        states = [board2sunfish(board, eval_pos(board, R)) for board in chess_boards]
        if use_player_move:
            player_moves_sunfish = [str_to_sunfish_move(move, not board.turn) for move, board in zip(player_moves, chess_boards)]
            actions = [max(move_dict, key=lambda k: move_dict[k][0] - (k == 'sum')) for move_dict in player_moves]
        else:
            actions = [sunfish_move(state, pst, config_data['time_limit'], True) for state in tqdm(states, desc='Getting true moves', )]
    else:
        raise Exception(f"The policy {config_data['chess_policy']} is not implemented yet")

    config_data['permute_idxs'] = np.array(char_to_idxs(config_data['permute_char']))
    R = np.array(config_data['R_start'], dtype=float)
    Rs = [R]
    start_time = time()

    # ============================================ Algorithm Start ==================================================
    # Define variables. If a veriable is modified, it is passed as input and overwritten by being assigned to the output, with the exception of bookkeeping().
    # Interpret following the policy as arriving at the same final board. 
    pi, a_pi, pi_new = [None] * len(states), [None] * len(states), [None] * len(states) 
    Qpi_policy_R    = np.zeros(len(states)) # Qpi(s,pi,R)
    Qpi_action_Rnew = np.zeros(len(states)) # Qpi(s,a ,R~)
    Qpi_policy_Rnew = np.zeros(len(states)) # Qpi(s,pi,R~)
    QpiNew_policy_Rnew = np.zeros(len(states)) # Qpi~(s,pi~,R~)
    accuracies = []
    energies = []
    Rs = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        
        # Calculate initially. 
        pi, Qpi_policy_R, pi_moves = PolicyIteration(states, pi, None, Qpi_policy_R, R, config_data, parallel)
        a_pi, _, _ = PolicyIteration(states, a_pi, actions, Qpi_policy_R, R, config_data, parallel) # We can't guarantee that alpha-beta search fully explores move a and so we calculate it again.
        bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R, Rs, R)

        for epoch in tqdm(range(config_data['epochs']), desc='Epochs'):
            weight_path = join(out_path, f'weights/{epoch}.csv')
            if os.path.exists(weight_path):
                df = pd.read_csv(weight_path)
                R = df['Result'].values.flatten()
                Rs.append(R)
                print(f'Results loaded for epoch {epoch+1}, continuing')
                continue # This messes up the initial calculated epoch a little bit. Fix later.
            
            R_new = perturb_reward(R, config_data, epoch)
            # Evaluate perterbued reward function
            Qpi_action_Rnew = Qeval(Qpi_action_Rnew, a_pi, states, R_new, parallel, pst = config_data['pst'])
            Qpi_policy_Rnew = Qeval(Qpi_policy_Rnew, pi, states, R_new, parallel, pst = config_data['pst']) # This the standard Q-value there, the policy should be optimal. 
            # np.corrcoef(Qpi_action_Rnew, Qpi_policy_Rnew) # We expect these to be highly correlated, as one move probably doesn't change much.  

            # Switch stochastically accept the new reward function and the new policy
            if np.any(Qpi_policy_Rnew < Qpi_action_Rnew): # if the new reward function explains the data action better than the policy action for any state
                pi_new, QpiNew_policy_Rnew, pi_new_moves = PolicyIteration(states, pi_new, None, QpiNew_policy_Rnew, R_new, config_data, parallel)
                log_prob = min(0, log_prob_dist(R_new, np.sum(QpiNew_policy_Rnew), alpha=config_data['alpha']) - log_prob_dist(R, np.sum(Qpi_policy_R), alpha=config_data['alpha']))

                if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                    print(f'Changed weights and policy! From {R}\n to {R_new}\n Probability was: {np.exp(log_prob)}')
                    R = np.copy(R_new)
                    pi, Qpi_policy_R, pi_moves = pi_new.copy(), np.copy(QpiNew_policy_Rnew), pi_new_moves.copy()
                    a_pi, _, _ = PolicyIteration(states, a_pi, actions, Qpi_policy_R, R, config_data, parallel)
            else:
                log_prob = min(0, log_prob_dist(R_new, np.sum(Qpi_policy_Rnew), alpha=config_data['alpha']) - log_prob_dist(R, np.sum(Qpi_policy_R), alpha=config_data['alpha']))
                if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                    print(f'Changed weights! From {R}\n to {R_new}\n Probability was: {np.exp(log_prob)}')
                    Qpi_policy_R = np.copy(Qpi_policy_Rnew)
                    R = np.copy(R_new)

            acc = bookkeeping(accuracies, actions, pi_moves, energies, Qpi_policy_R, Rs, R)
            process_epoch(R, epoch, config_data, out_path)
            print(f'Current sunfish accuracy: {acc}, best: {max(accuracies)}')
            print(f'Best R: {R}')
            if time() - start_time > config_data['max_hours'] * 60 * 60:
                break

            if ((epoch + 1) % config_data['val_every']) == 0 or (epoch + 1) == config_data['epochs']:
                pst_val = get_new_pst(R)
                val_util(validation_set, out_path, config_data, parallel, pst_val, name=epoch)
        return accuracies