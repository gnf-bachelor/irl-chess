import os
from os.path import join
from irl_chess.misc_utils.utils import union_dicts
from irl_chess.misc_utils.load_save_utils import fix_cwd, load_config, create_result_path, get_states

if __name__ == '__main__':
    fix_cwd()
    base_config_data, model_config_data = load_config()
    config_data = union_dicts(base_config_data, model_config_data)

    match config_data['model']: # Load the model specified in the "base_config" file. Make sure the "model" field is set 
                                # correctly and that a model_result_string function is defined to properly store the results.
        case "sunfish_permutation_native":
            from irl_chess.sunfish_native_pw import run_sunfish_native as model, \
                                          sunfish_native_result_string as model_result_string
        case "bayesian_optimisation":
            from irl_chess.bayesian_optimisation import run_bayesian_optimisation as model, \
                                          bayesian_model_result_string as model_result_string

        case _ :
            raise Exception(f"No model found with the name {config_data['model']}")

    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)

    websites_filepath = join(os.getcwd(), 'downloads', 'lichess_websites.txt')
    file_path_data = join(os.getcwd(), 'data', 'raw')

    sunfish_boards = get_states(websites_filepath=websites_filepath,
                                file_path_data=file_path_data,
                                config_data=config_data) # Boards in the sunfish format.
    
    model(sunfish_boards=sunfish_boards, config_data=config_data, out_path=out_path)