import os
import json
from os.path import join

if __name__ == '__main__':
    if os.getcwd()[-len('irl-chess'):] != 'irl-chess':
        print(os.getcwd())
        os.chdir('../')
    from irl_chess import plot_permuted_sunfish_weights, create_sunfish_path

    with open(join(os.getcwd(), 'experiment_configs', 'current', 'config.json'), 'r') as file:
        config_data = json.load(file)

    path_result = join(os.getcwd(), 'models', 'sunfish_permuted')
    out_path = create_sunfish_path(config_data=config_data, path_result=path_result)
    plot_permuted_sunfish_weights(config_data=config_data, out_path=out_path)
