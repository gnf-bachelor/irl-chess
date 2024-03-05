import os
import json
from os.path import join
def vscode_fix(): # How do I make this function available everywhere? It is really annoying. 
    if 'TERM_PROGRAM' in os.environ.keys() and os.environ['TERM_PROGRAM'] == 'vscode':
        print("Running in VS Code, fixing sys path")
        import sys

        sys.path.append("./")

if __name__ == '__main__':
    vscode_fix()

    from project import plot_permuted_sunfish_weights, create_sunfish_path

    with open(join(os.getcwd(), 'experiment_configs', 'sunfish_permutation_native', 'config.json'), 'r') as file:
        config_data = json.load(file)

    n_files = config_data['n_files']
    overwrite = config_data['overwrite']
    version = config_data['version']

    path_result = join(os.getcwd(), 'models', 'sunfish_permuted_native')
    out_path = create_sunfish_path(config_data, path_result)

    plot_permuted_sunfish_weights(config_data, out_path)
