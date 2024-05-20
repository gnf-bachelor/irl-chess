import os
from os.path import join
from time import time

import numpy as np
import pandas as pd
import scipy
from joblib import delayed, Parallel
from tqdm import tqdm

from irl_chess.maia_chess import load_maia_network


def maia_pre_move(state, model, topk=1):
    """
    Given a state, p-square table and time limit,
    return the sunfish move.
    :param state:
    :param pst:
    :param time_limit:
    :return:
    """
    move_and_score =  model.getTopMovesCP(state, topk)[0]
    return move_and_score


def maia_pre_result_string(model_config_data):
    time_limit = model_config_data['time_limit']
    maia_elo = model_config_data['maia_elo']
    topk = model_config_data['topk']
    return f"{time_limit}-{maia_elo}-{topk}"


def run_maia_pre(chess_boards, player_moves, config_data, out_path, validation_set, model=None, return_model=False):
    start_time = time()
    model = load_maia_network(elo=config_data['maia_elo'],
                              time_limit=config_data['time_limit'],
                              parent=join('irl_chess', 'maia_chess')) if model is None else model

    actions_val = [maia_pre_move(state, model) for state, move in
                   tqdm(validation_set, desc='Getting Maia validation actions')]

    acc_temp = []
    data_save = []
    for (state, a_true), (a_val, score) in zip(validation_set, actions_val):
        a_true = str(a_true)
        a_val = str(a_val)
        acc_temp.append(a_true == a_val)
        data_save.append((state, a_true, a_val, score))
    acc = sum(acc_temp) / len(acc_temp)
    print(f'Maia validation accuracy: {acc}')
    df = pd.DataFrame(data_save)
    os.makedirs(out_path, exist_ok=True)
    df.to_csv(join(out_path, 'validation_output.csv'))
    print(f'Finished getting Maia moves in {time() - start_time:.3f} seconds')
    if return_model:
        from irl_chess import wilson_score_interval
        return acc, model, wilson_score_interval(sum(acc_temp), len(acc_temp))
    return acc
