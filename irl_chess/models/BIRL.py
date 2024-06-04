from time import time
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
import logging
from irl_chess.misc_utils.load_save_utils import process_epoch, load_Rs, load_previous_results, save_array
from irl_chess.chess_utils.sunfish_utils import board2sunfish, get_new_pst, str_to_sunfish_move, eval_pos, sunfish_move, eval_pos_pst
from irl_chess.chess_utils.BIRL_utils import pi_alpha_beta_search, pi_alpha_beta_search_par, \
    Qeval_chessBoard, Qeval_chessBoard_par, sunfish_search_par, bookkeeping, perturb_reward, log_prob_dist, Qeval_sunfishBoard_par, eval_log_prob
from irl_chess.chess_utils.sunfish import pst


def BIRL_result_string(model_config_data):
    energy_optimized = model_config_data['energy_optimized']
    chess_policy = model_config_data['chess_policy']
    delta = model_config_data['delta']
    decay = model_config_data['decay']
    decay_step = model_config_data['decay_step']
    noise_distribution = model_config_data['noise_distribution']
    if chess_policy == "alpha_beta":
        depth = model_config_data['depth']
        quisce = model_config_data["quiesce"]
        return f"{energy_optimized}-{chess_policy}-{noise_distribution}-{int(delta)}-{decay}-{decay_step}-{depth}-{quisce}"
    elif chess_policy == "sunfish":
        time_limit = model_config_data['time_limit']
        return f"{energy_optimized}-{chess_policy}-{noise_distribution}-{int(delta)}-{decay}-{decay_step}-{time_limit}"
    

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
        from irl_chess.chess_utils.BIRL_utils import sunfish_search_par as PolicyIteration, \
                Qeval_sunfishBoard_par as Qeval
        states = [board2sunfish(board, eval_pos_pst(board, pst)) for board in chess_boards] # sunfish scores are relative, so they have no bearing. 
        actions = [str_to_sunfish_move(move, not board.turn) for move, board in zip(player_moves, chess_boards)]
    else:
        raise Exception(f"The policy {config_data['chess_policy']} is not implemented yet")
    assert config_data['energy_optimized'] in ["policy", "action"], "Energy optimized must be either policy or action"
    
    RPs, Rpsts, RHs, next_empty_epoch = load_previous_results(out_path)
    if next_empty_epoch == 0:
        RP, Rpst, RH = load_Rs(config_data)
    else:
        RP, Rpst, RH = RPs[-1], Rpsts[-1], RHs[-1]
    start_time = time()
    # ============================================ Algorithm Start ==================================================
    # Define variables for clarity. All other functions are functional and do not modify their inputs.
    # Interpret following the policy as arriving at the same final board. 
    pi, a_pi, pi_new = [None] * len(states), [None] * len(states), [None] * len(states) 
    Qpi_policy_R    = np.zeros(len(states)) # Qpi(s,pi,R)
    Qpi_action_Rnew = np.zeros(len(states)) # Qpi(s,a ,R~)
    Qpi_policy_Rnew = np.zeros(len(states)) # Qpi(s,pi,R~)
    QpiNew_policy_Rnew = np.zeros(len(states)) # Qpi~(s,pi~,R~)
    accuracies = []
    pi_energies = []
    a_energies = []
    # RPs, Rpsts, RHs = [], [], []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        
        # Calculate initially.
        print("Calculating initial policy.") 
        pi, Qpi_policy_R, pi_moves = PolicyIteration(states, None, RP, Rpst, RH, config_data, parallel)
        print("Calculating initial action policy.")
        a_pi, Qpi_action_R, _ = PolicyIteration(states, actions, RP, Rpst, RH, config_data, parallel) # We can't guarantee that alpha-beta search fully explores move a and so we calculate it again.
        bookkeeping(accuracies, actions, pi_moves, pi_energies, a_energies, Qpi_policy_R, Qpi_action_R, RPs, RP, Rpsts, Rpst, RHs, RH) # Bookkeeping for the initialization.
        print("Finished initial policy calculation.")

        for epoch in tqdm(range(next_empty_epoch, config_data['epochs']), desc='Epochs'):
            
            RP_new, Rpst_new, RH_new = perturb_reward(RP, config_data, Rpst=Rpst, RH=RH, epoch = epoch) # Also handles delta decay
            # Evaluate perterbued reward function
            Qpi_action_Rnew = Qeval(a_pi, states, RP_new, Rpst_new, RH_new, parallel)
            Qpi_policy_Rnew = Qeval(pi, states, RP_new, Rpst_new, RH_new, parallel) # This the standard Q-value there, the policy should be optimal. 
            # np.corrcoef(Qpi_action_Rnew, Qpi_policy_Rnew) # We expect these to be highly correlated, as one move probably doesn't change much.  

            # Switch stochastically accept the new reward function and the new policy
            if np.any(Qpi_policy_Rnew < Qpi_action_Rnew): # if the new reward function explains the data action better than the policy action for any state
                pi_new, QpiNew_policy_Rnew, pi_new_moves = PolicyIteration(states, None, RP_new, Rpst_new, RH_new, config_data, parallel)
                a_pi_new, QpiNew_action_Rnew, _ = PolicyIteration(states, actions, RP_new, Rpst_new, RH_new, config_data, parallel)
                if config_data['energy_optimized'] == "policy":
                    log_prob = eval_log_prob(RP_new, QpiNew_policy_Rnew, RP, Qpi_policy_R, config_data['alpha'])
                elif config_data['energy_optimized'] == "action":
                    log_prob = eval_log_prob(RP_new, QpiNew_action_Rnew, RP, Qpi_action_R, config_data['alpha'])
                if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                    print(f'Changed weights and policy! From {RP}\n to {RP_new}\n Probability was: {np.exp(log_prob)}')
                    RP, Rpst, RH = RP_new.copy(), Rpst_new.copy(), RH_new.copy()
                    pi, Qpi_policy_R, pi_moves = pi_new.copy(), np.copy(QpiNew_policy_Rnew), pi_new_moves.copy()
                    a_pi, Qpi_action_R = a_pi_new.copy(), np.copy(QpiNew_action_Rnew)

            else:
                if config_data['energy_optimized'] == "policy":
                    log_prob = eval_log_prob(RP_new, Qpi_policy_Rnew, RP, Qpi_policy_R, config_data['alpha'])
                elif config_data['energy_optimized'] == "action":
                    log_prob = eval_log_prob(RP_new, Qpi_action_Rnew, RP, Qpi_action_R, config_data['alpha'])
                if log_prob > -1e3 and np.random.random() < np.exp(log_prob):
                    print(f'Changed weights! From {RP}\n to {RP_new}\n Probability was: {np.exp(log_prob)}')
                    Qpi_policy_R = np.copy(Qpi_policy_Rnew)
                    RP, Rpst, RH = RP_new.copy(), Rpst_new.copy(), RH_new.copy()

            acc = bookkeeping(accuracies, actions, pi_moves, pi_energies, a_energies, Qpi_policy_R, Qpi_action_R, RPs, RP, Rpsts, Rpst, RHs, RH)
            process_epoch(RP, Rpst, RH, epoch, config_data, out_path)
            print(f'Current sunfish accuracy: {acc}, best: {max(accuracies)}')
            print(f'Best R: {RP}')
            if time() - start_time > config_data['max_hours'] * 60 * 60:
                break

            # Maybe validate
            # if ((epoch + 1) % config_data['val_every']) == 0 or (epoch + 1) == config_data['epochs']:
            #     pst_val = get_new_pst(R)
            #     val_util(validation_set, out_path, config_data, parallel, pst_val, name=epoch)
        save_array(accuracies, "temp_accuracies", out_path)
        save_array(pi_energies, "pi_energies", out_path)
        save_array(a_energies, "a_energies", out_path)
        return accuracies