
import os
from os.path import join
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from irl_chess.visualizations import *


def load_single_csv(out_path, name):
    path = os.path.join(out_path, f'weights/{name}.csv')
    if os.path.exists(path):
        return pd.read_csv(path, index_col=None)
    else: return None

def plot_seaborn_weights(config_data, out_path, start_weight_idx=0, epoch=None, show=False, filename_addition='', **kwargs):
    # Load weights
    weights = load_weights(out_path, 'Result', start_weight_idx=start_weight_idx, epoch=epoch)

    plot_path = os.path.join(out_path, 'plots', 'RP')
    os.makedirs(plot_path, exist_ok=True)

    # Load true weights if available
    weights_true = np.array(config_data.get('RP_true', []))
    
    # Ensure plot_char is a subset of the allowed pieces
    allowed_pieces = ['P', 'N', 'B', 'R', 'Q', 'K']
    plot_char = config_data.get('plot_char', allowed_pieces)
    plot_char = [char for char in plot_char if char in allowed_pieces]

    if not plot_char:
        raise ValueError("plot_char must contain at least one valid chess piece character from ['P', 'N', 'B', 'R', 'Q', 'K'].")

    # Filter the weights data to include only the specified pieces
    piece_indices = [allowed_pieces.index(char) for char in plot_char]
    filtered_weights = weights[:, piece_indices]
    filtered_weights_true = weights_true[piece_indices] if len(weights_true) > 0 else []


    # Prepare data for plotting
    epochs = np.arange(start_weight_idx, start_weight_idx + len(filtered_weights))
    weights_df = pd.DataFrame(filtered_weights, columns=plot_char)
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
    if len(filtered_weights_true) > 0:
        for piece, true_weight in zip(plot_char, filtered_weights_true):
            plt.axhline(y=true_weight, color=palette[plot_char.index(piece)], linestyle='--', linewidth=1.5)

    # Customize the plot
    plt.title(f'Sunfish Weights Over Time for ELO {config_data["min_elo"]}-{config_data["max_elo"]}')
    plt.xlabel('Epochs')
    plt.ylabel('Weight Values')
    plt.legend(title='Piece', loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    
    # Save the plot
    plot_path = join(plot_path, f'weights_seaborn_{epoch}{filename_addition}.svg')
    plt.savefig(plot_path)
    if show:
        plt.show()
    plt.close()

def plot_accuracies_and_energies(out_path, show=False, filename_addition=''):
    """Plot accuracies and energies from two CSV files on a single graph."""
    plot_path = os.path.join(out_path, 'plots', 'acc')
    os.makedirs(plot_path, exist_ok=True)
    
    # Load data
    accuracies_df = load_single_csv(out_path, 'temp_accuracies')
    energies_df = load_single_csv(out_path, 'energies')

    
    # Check if the data frames are non-empty and have the correct format
    if accuracies_df.empty or energies_df.empty:
        raise ValueError("One or both of the data files are empty.")
        
    
    if 'temp_accuracies' not in accuracies_df.columns:
        raise ValueError("Accuracy CSV must contain an 'temp_accuracies' column.")
    else:
        accuracies_df.rename(columns={'temp_accuracies': 'Accuracy'}, inplace=True)
    
    if 'energies' not in energies_df.columns:
        raise ValueError("Energy CSV must contain an 'energies' column.")
    else:
        energies_df.rename(columns={'energies': 'Energy'}, inplace=True)
    
    # Create an 'Epoch' column based on the index
    accuracies_df['Epoch'] = accuracies_df.index
    energies_df['Epoch'] = energies_df.index
    
    # Merge the two dataframes on the 'Epoch' column
    merged_df = pd.merge(accuracies_df, energies_df, on='Epoch', how='inner')

    # Initialize the seaborn plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.set(style="whitegrid")

    # Plot accuracies on the primary y-axis
    sns.lineplot(data=merged_df, x='Epoch', y='Accuracy', color='blue', linewidth=2.5, ax=ax1, label='Accuracy')

    # Plot energies on the secondary y-axis
    ax2 = ax1.twinx()
    sns.lineplot(data=merged_df, x='Epoch', y='Energy', color='red', linewidth=2.5, ax=ax2, label='Energy')

    # Customize the plot
    ax1.set_title('Accuracies and Energies Over Time')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy', color='blue')
    ax2.set_ylabel('Energy', color='red')

    # Align the legends
    ax1.legend(loc='upper left', bbox_to_anchor=(0, 1), borderaxespad=0.)
    ax2.legend(loc='upper right', bbox_to_anchor=(1, 1), borderaxespad=0.)

    fig.tight_layout()

    # Save the plot
    plt.savefig(join(plot_path, f'accuracies_energies_plot{filename_addition}.svg'))
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
    plot_accuracies_and_energies(out_path, show=True, filename_addition='')