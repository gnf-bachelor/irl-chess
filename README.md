# chess-irl
A project exploring approximate Inverse Reinforcement Learning (IRL) applications in Chess. 
This includes a modified version of the Bayesian Inverse Reinforcement Learning (BIRL) algortihm "Policy Walk", and our own more naive method, "Greedy Policy Walk" (GPW) designed to work for the large state space of Chess.

| Trace Plot of GPW Piece Weights on 1900-2000 ELO Player Moves | Trace Plot of GPW PST Weights on 1900-2000 ELO Player Moves |
|:------------------------------:|:------------------------------:|
| ![trace_plot_GPW_1900_player_move_piece_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/8d6246a2-ff46-41b4-a4c3-30564cccfc5a) | ![trace_plot_GPW_1900_player_move_pst_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/e032689a-a006-4d5d-82df-83f4faed56a8) |

| Trace Plot of GPW Piece Weights on Synthetic Engine Moves | Trace Plot of GPW PST Weights on Synthetic Engine Moves|
|:------------------------------:|:------------------------------:|
| ![trace_plot_GPW_1900_sunfish_move_piece_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/0312216b-3df7-43dc-9fcc-5dd0f371a865)  | ![trace_plot_GPW_1900_sunfish_move_pst_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/470db9fe-f82d-4dec-9b61-e28392935e20)|

## Project structure
The repository contains a modular setup for running the various algorithms with a single entry-point (run_model.py). All settings, including which model to use, are set in config files base_config.json and the model specific config.json.

![Pipeline](https://github.com/gnf-bachelor/irl-chess/assets/98162688/58f71093-a4ba-4f54-a9d4-a824788ff5c6)

The directory structure of the project looks like this:

```txt

├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── hpc-logs        <- Folder for output when running on the DTU cluster
│
├── hpc-submit      <- Shell scripts for running GPW on the DTU cluster
│
├── experiment_configs      <- json config files for each of the models
│
├── results      <- folder where results from running the models end up
│
├── irl_chess  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download and filter data
│   │
│   ├── maia_chess           <- maia chess repository with some modifications
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │
│   ├── chess_utils    <- Scripts to work with chess and the search algorithms
│   │
│   ├── misc_utils    <- Scripts that don't neatly fit into any other category
│   │
│   ├── models    <- Scripts for each of the implemented models
│   │
│   ├── stat_tools    <- Scripts for confidence intervals
│   │
│   ├── thesis_figures    <- Scripts generating figures found in the thesis
│   │
│   ├── run_model.py   <- script for training the model
```

## How to Run BIRL or GPW

First, setup your python environment. Make sure to use python 3.10 for all dependencies and submodules to work properly. You can either install the `conda` environment, [`irl_chess_env.yml`](irl_chess_env.yml), or you can make sure all the required packages are installed from `requirements.txt`. Then run `pip install .` to install `irl_chess` as a python package. 

Then adjust the config files.
To select the desired model change the "model" parameter in the base_config.json in the experiment_config folder.
Each model has a folder containing its own config where parameters specific to the model are set. See the Config Description section.

When parameters in the base and model-configs are set to the desired values, simply run run_model.py
The results will be saved the results folder under a name that indicates what parameters where used for the run.

Computation wise, running Greedy Policy Walk for 300 epochs evaluated on 10,000 boards with 24 CPU cores and a time limit of 0.2 seconds takes about 4 hours.

## How to build Maia chess components
I you wish to use the maia-chess comparison functionalities, after cloning the repository locally, don't forget to run the following commands in order to also clone the git submodules:

```
git submodule sync --recursive
git submodule update --recursive --init
```

Then go to https://lczero.org/play/download/ and download an appropriate backend. Place the contents in
a subfolder of maia-chess called 'lc0-exe-folder' (maia-chess/lc0-exe-folder). 

Run pip install -r maia_chess/requirements.txt

Now you can use maia_chess models using the following:

from maia_chess import load_maia_network

model = load_maia_network(1100, parent='maia_chess/')

This script keeps running in the background until stopped externally.

# Config Description

## base_config.json

### General Settings
- **overwrite**: `false`  
  Whether to redownload and overwrite existing data or not.

- **n_files**: `1`  
  Number of files to process.

- **time_control**: `false`  
  Whether to use time control in the game.

### ELO Rating
- **min_elo**: `1900`  
  Minimum ELO rating of the players to consider (inclusive).

- **max_elo**: `2000`  
  Maximum ELO rating of the players to consider (noninclusive).

### Game Parameters
- **n_midgame**: `10`  
  The move ply at which to begin sampling positions (inclusive).

- **n_endgame**: `100`  
  The move ply at which to stop sampling positions (inclusive).

- **n_boards**: `10000`  
  Number of boards to use for training.

- **n_boards_val**: `200`  
  Number of boards to use for validation.

### Training Parameters
- **epochs**: `300`  
  Number of epochs for training.

- **max_hours**: `100`  
  Maximum number of hours to run the training.

- **n_threads**: `-1`  
  Number of threads to use for processing (-1 to use all available).

### Evaluation Parameters
- **plot_every**: `10`  
  Frequency (in epochs) to plot the results.

- **val_every**: `10`  
  Frequency (in epochs) to perform validation.

- **run_n_times**: `1`  
  Number of times to run the entire process (for trace plots).

### Weight Permutation and Initialization.
Here piece names are abbreviated using their typical chess letter. 
P: Pawn, N: Knight, B: Bishop, R: Rook, Q: Queen, K: King. 

### Piece Value Plotting and Permutation
- **plot_char**: `["P", "N", "B", "R", "Q"]`  
  Chess piece value weights to plot during the evaluation.

- **permute_char**: `["N", "B", "R", "Q"]`  
  Chess piece value weights to permute during the evaluation.

- **RP_start**: `[100, 100, 100, 100, 100, 60000]`  
  Starting values for piece weights. RP: Reward function Piece weights.  

### Piece-Square Table (PST) Plotting and Permutation
- **plot_pst_char**: `["P", "N", "B", "R", "Q", "K"]`  
  Piece square table weights to plot during the evaluation.

- **permute_pst_char**: `["P", "N", "B", "R", "Q", "K"]`  
  Piece square table weights to permute during the evaluation.

- **Rpst_start**: `[0, 0, 0, 0, 0, 0]`  
  Starting values for piece square table weights. Rpst: Reward function Piece-Square Table weights.

### Additional Settings
- **include_PA_KS_PS**: `[false, false, false]`  
  Include Piece Activity (PA), King Safety (KS), and Pawn Structure (PS) heuristics in the evaluation. Currently only implemented with the alpha-beta searcher. 

- **plot_H**: `[true, true, true]`  
  Whether to plot heuristic parameters during the evaluation.

- **permute_H**: `[true, true, true]`  
  Whether to permute heuristic parameters during the evaluation.

- **RH_start**: `[0, 0, 0]`  
  Starting values for heuristic weights.

### Move Function and Model
- **move_function**: `"player_move"`  
  The function used to sample moves from. Options are "player_move" or "sunfish_move" for synthetic engine moves. 

#### True weights in the case of synthetic "Sunfish" engine moves
- **RP_true**: `[100, 280, 320, 479, 929, 60000]`  
  True values for RP for each piece. 

- **Rpst_true**: `[1, 1, 1, 1, 1, 1]`  
  True values for Rpst for each piece square table.

- **RH_true**: `[0, 0, 0]`  
  True values for RH for each heuristic.


- **permute_how_many**: `1`  
  Number of weights to permute in each iteration ("all" or "-1" permutes all weights).

- **model**: `"sunfish_GRW"`  
  The model to use for evaluation. Options include "sunfish_GRW" (GPW), "BIRL" and "maia_pretrained".

## config.json
The config.json files contain model specific settings. Here the BIRL model config file is showcased as an example. 

### BIRL Configuration Parameters

- **energy_optimized**: `"action"`  
  Whether the model seeks to optimize the energy of all actions in the state action pairs in the data set ("action"), or the energy of the policy moves in all states in the data set ("policy"). 

- **alpha**: `1e-1`  
  The alpha "inverse temperature" parameter value for the boltzman energy distribution. The lower alpha, the less likely the agent is to make the optimal decision.

- **chess_policy**: `"sunfish"`  
  The policy used for making decisions in chess ("sunfish" or "alpha_beta").

#### Alpha-Beta Parameters
- **depth**: `3`  
  The depth of the search tree.

- **quiesce**: `false`  
  Whether to use quiescence search to avoid horizon effects.

#### Sunfish Parameters
- **time_limit**: `0.2`  
  The time allotted to search for a move in seconds.

#### Weight Pertubation Parameters
- **noise_distribution**: `"gaussian"`  
  The distribution of noise added to the weights in each iteration ("gaussian", "step" or "uniform").

- **delta**: `40`  
  A parameter describing the variance of the noise. The standard deviation in the case of Gaussian noise, the step size for "step" and the domain [-delta, delta] for "uniform". 

- **decay**: `1`  
  The multiplicative decay rate for delta.

- **decay_step**: `1000`  
  Frequency at which decay is applied in epochs.

## Additional Configuration Parameters
The configs of "sunfish_GRW" are a subset of "BIRL", and the two "maia_pretrained" specific configs are shown below. 

- **maia_elo**: `1900`  
  The ELO rating of the Maia model used.

- **topk**: `1`  
  The top K moves to consider when evaluating accuracy.

