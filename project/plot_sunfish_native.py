import os
import json
from os.path import join


if __name__ == '__main__':
    from project import plot_permuted_sunfish_weights

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation_native', 'config.json'), 'r') as file:
        config_data = json.load(file)

    n_files = config_data['n_files']
    overwrite = config_data['overwrite']
    version = config_data['version']

    out_path = join(os.getcwd(), 'models', 'sunfish_permuted_native')
    plot_permuted_sunfish_weights(config_data, out_path)
