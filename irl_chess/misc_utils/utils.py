import os
from os.path import join

import numpy as np
import pandas as pd


def union_dicts(dict1, dict2):
    # Check for common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())
    if common_keys:
        raise ValueError(f"Error: Dictionaries have common keys: {common_keys}")

    # If no common keys, perform the union
    return {**dict1, **dict2}

def create_default_sunfish(out_path, base_config):
    default_weight_path = join(out_path, 'weights')
    os.makedirs(default_weight_path, exist_ok=True)
    R_default = np.array(base_config['RP_true'])
    R_pst = np.array(base_config['Rpst_true'])
    df = pd.DataFrame(np.concatenate((R_default.reshape((-1, 1)), R_pst.reshape((-1, 1))), axis=1),
                      columns=['Result', 'RpstResult'])
    df.to_csv(join(default_weight_path, '0.csv'))
    print(f'Created sunfish weights at {join(default_weight_path, "0.csv")}')


def reformat_list(lst, inbetween_char = ''):
    return inbetween_char.join(map(str, lst))