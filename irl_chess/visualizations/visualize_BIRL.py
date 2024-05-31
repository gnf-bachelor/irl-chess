
import os
from os.path import join
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from irl_chess.visualizations import *

def plot_seaborn_weights(config_data, out_path, start_weight_idx=0, epoch=None, show=False, filename_addition=''):
    # Load weights
    weights = load_weights(out_path, 'Result', start_weight_idx=start_weight_idx, epoch=epoch)

    plot_path = os.path.join(out_path, 'plots', 'RP')
    os.makedirs(plot_path, exist_ok=True)

    # Load true weights if available
    weights_true = np.array(config_data.get('true_weights', []))
    
    # Prepare data for plotting
    epochs = np.arange(start_weight_idx, epoch)
    plot_char = ['P', 'N', 'B', 'R', 'Q', 'K']
    # plot_char = config_data.get('plot_char', ['P', 'N', 'B', 'R', 'Q', 'K'])  # Assuming these are the pieces being tracked

    weights_df = pd.DataFrame(weights, columns=plot_char)
    weights_df['Epoch'] = epochs
    
    # Create a melt dataframe suitable for seaborn
    melt_df = weights_df.melt(id_vars=['Epoch'], var_name='Piece', value_name='Weight')
    
    # Initialize the seaborn plot
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    palette = sns.color_palette(sunfish_palette_name, n_colors=len(plot_char))
    
    # Create the line plot
    sns.lineplot(data=melt_df, x='Epoch', y='Weight', hue='Piece', palette=palette, linewidth=2.5)
    
    # Add horizontal lines for true weights if available
    if len(weights_true) > 0:
        for piece, true_weight in zip(plot_char, weights_true):
            plt.axhline(y=true_weight, color=palette[plot_char.index(piece)], linestyle='--', linewidth=1.5)
    
    # Customize the plot
    plt.title(f'Sunfish Weights Over Time for ELO {config_data["min_elo"]}-{config_data["max_elo"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Values')
    plt.legend(title='Piece', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Save the plot
    plot_path = join(out_path, f'weights_seaborn_{epoch}{filename_addition}.png')
    plt.savefig(plot_path)
    if show:
        plt.show()
    plt.close()


if __name__ == '__main__':
    from irl_chess import fix_cwd, load_config, load_model_functions, union_dicts, create_result_path
    fix_cwd()
    base_config_data, model_config_data = load_config()
    model, model_result_string = load_model_functions(base_config_data)
    config_data = union_dicts(base_config_data, model_config_data)
    out_path = create_result_path(base_config_data, model_config_data, model_result_string, path_result=None)
    assert os.path.isdir(out_path), 'The result path does not exist, which means that result data has not been generated for this configuration.\
          Run run_model with the correct configs to generate results and plots.'
    
    for epoch in range(config_data['epochs']): # Check how many epochs the weights have been saved for.
        path = os.path.join(out_path, f'weights/{epoch}.csv')
        if os.path.exists(path):
            epoch += 1
        else:
            epoch -= 1
            print("Weights have been saved for", epoch+1, "epochs.")
            break

    plot_seaborn_weights(config_data, out_path, start_weight_idx=0, epoch=epoch, show=True)