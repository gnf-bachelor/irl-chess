import copy
import os
import pickle
from os.path import join
from time import time

import chess
import chess.pgn
import chess.svg
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from irl_chess import Searcher, pst, piece, plot_R_weights, perturb_reward
from irl_chess.chess_utils.sunfish_utils import board2sunfish, sunfish2board, sunfish_move_to_str, \
    check_moved_same_color, sunfish_move
from irl_chess.visualizations import char_to_idxs

from irl_chess.misc_utils.utils import reformat_list
from irl_chess.misc_utils.load_save_utils import process_epoch
from irl_chess.chess_utils.sunfish_utils import get_new_pst, str_to_sunfish_move, check_moved_same_color, eval_pos, eval_pos_pst
from irl_chess.stat_tools.stat_tools import wilson_score_interval

def sunfish_native_result_string(model_config_data):
    delta = model_config_data['delta']
    decay = model_config_data['decay']
    decay_step = model_config_data['decay_step']
    time_limit = model_config_data['time_limit']
    noise_distribution = model_config_data['noise_distribution']
    return f"{noise_distribution}-{int(delta)}-{decay}-{decay_step}-{time_limit}"


def run_sunfish_GRW(chess_boards, player_moves, config_data, out_path, validation_set):
    if config_data['move_function'] == "sunfish_move":
        use_player_move = False
    elif config_data['move_function'] == "player_move":
        use_player_move = True
    else:
        raise Exception(f"The move function {config_data['move_function']} is not implemented yet")

    RP = np.array(config_data['RP_start'], dtype=float)
    Rpst = np.array(config_data['Rpst_start'], dtype=float)
    RH = np.array(config_data['RH_start'], dtype=float) 
    RH[np.invert(config_data['include_PA_KS_PS'])] = 0 # Set the values to 0 if they are not included in the model.
    pst_init = get_new_pst(RP=RP, Rpst=Rpst)
    Rs, Rpsts, RHs = [RP], [Rpst], [RH]
    sunfish_boards = [board2sunfish(board, eval_pos_pst(board, pst_init)) for board in chess_boards]
    player_moves_sunfish = [str_to_sunfish_move(move, not board.turn) for move, board in
                            zip(player_moves, chess_boards)]
    
    start_time = time()
    last_acc = 0
    accuracies = []
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        if use_player_move:
            actions_true = player_moves_sunfish
        else:
            pst_true = get_new_pst(RP=config_data['RP_true'], Rpst=config_data['Rpst_true'])
            actions_true = parallel(delayed(sunfish_move)(state, pst_true, config_data['time_limit'], True)
                for state in tqdm(sunfish_boards, desc='Getting true Sunfish moves', ))
        for epoch in tqdm(range(config_data['epochs']), desc='Epochs'):
            weight_path = join(out_path, f'weights/{epoch}.csv')
            if os.path.exists(weight_path):
                df = pd.read_csv(weight_path)
                RP = df['Result'].values.flatten()
                Rs.append(RP)
                print(f'Results loaded for epoch {epoch + 1}, continuing')
                continue
            
            RP_new, Rpst_new, RH_new = perturb_reward(RP, config_data, Rpst=Rpst, RH=RH, epoch = epoch) # Also handles delta decay

            pst_new = get_new_pst(RP_new, Rpst_new)  # Sunfish uses only pst table for calculations
            actions_new = parallel(delayed(sunfish_move)(board, pst_new, config_data['time_limit'], True)
                                   for board in tqdm(sunfish_boards, desc='Getting new Sunfish actions'))
            
            # check sunfish moves same color as player
            check_moved_same_color(sunfish_boards, player_moves_sunfish, actions_new)

            acc = sum([a == a_new for a, a_new in zip(actions_true, actions_new)]) / config_data['n_boards']
            if acc >= last_acc:
                RP, Rpst, RH = RP_new.copy(), Rpst_new.copy(), RH_new.copy()
                last_acc = copy.copy(acc)

            accuracies.append((acc, last_acc))
            Rs.append(RP), Rpsts.append(Rpst), RHs.append(RH)

            process_epoch(RP, Rpst, RH, epoch, config_data, out_path)

            print(f'Current sunfish accuracy: {acc}, best: {last_acc}')
            print(f'Best RP: {RP}')
            print(f'Best Rpst: {Rpst}')
            if time() - start_time > config_data['max_hours'] * 60 * 60:
                print(f'Reached time limit, exited at epoch {epoch}')
                break

            if (epoch + 1) % config_data['val_every'] == 0:
                pst_val = get_new_pst(RP, Rpst)
                val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves=use_player_move, name=epoch)
        pst_val = get_new_pst(RP, Rpst)
        out = val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves=use_player_move, name=epoch)
        return out


def val_sunfish_GRW(validation_set, out_path, config_data, epoch, use_player_moves, name):
    with (Parallel(n_jobs=config_data['n_threads']) as parallel):
        df = pd.read_csv(join(out_path, 'weights', f'{epoch}.csv'))
        RP = df['Result'].values.flatten()
        Rpst = df['RpstResult'].values.flatten()
        pst_val = get_new_pst(RP, Rpst)
        return val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves, name)


def val_util(validation_set, out_path, config_data, parallel, pst_val, use_player_moves, name):
    csv_path = join(out_path, 'validation_output', f'{name}.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        actions_val = list(df['a_val'])
        actions_true = list(df['a_true'])
    else:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        actions_val = parallel(
            delayed(sunfish_move)(board2sunfish(board, eval_pos(board, None)), pst_val, config_data['time_limit'], True)
            for board, move in tqdm(validation_set, desc='Getting True Sunfish actions'))
        actions_true = parallel(
            delayed(sunfish_move)(board2sunfish(board, eval_pos(board, None)), pst, config_data['time_limit'], True)
            for board, move in tqdm(validation_set, desc='Getting Sunfish validation actions')) if not use_player_moves else actions_val

    acc_temp_true, acc_temp_player = [], []
    actions_val_san, actions_true_san = [], []

    for (state, a_player), a_val, a_true in zip(validation_set, actions_val, actions_true):
        a_val = a_val if type(a_val) is str else sunfish_move_to_str(a_val, not state.turn)
        a_true = a_true if type(a_true) is str else sunfish_move_to_str(a_true, not state.turn)

        actions_val_san.append(a_val)
        actions_true_san.append(a_true)

        acc_temp_true.append(str(a_true) == a_val)
        acc_temp_player.append(str(a_player) == a_val)

    acc_true = sum(acc_temp_true) / len(acc_temp_true)
    acc_player = sum(acc_temp_player) / len(acc_temp_player)
    if not use_player_moves:
        print(f'Sunfish Validation accuracy on Sunfish: {acc_true}')
    print(f'Sunfish Validation accuracy on player moves: {acc_player}')

    df = pd.DataFrame([(state, a_player, a_true, a_val) for (state, a_player), a_val, a_true in
                       zip(validation_set, actions_val_san, actions_true_san)],
                      columns=['board', 'a_player', 'a_true', 'a_val'])
    df.to_csv(csv_path, index=False)
    return acc_player, wilson_score_interval(sum(acc_temp_player), len(acc_temp_player))
