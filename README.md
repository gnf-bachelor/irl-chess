# chess-irl
A project exploring approximate Inverse Reinforcement Learning (IRL) applications in Chess. 
This includes a modified version of the Bayesian Inverse Reinforcement Learning (BIRL) algortihm "Policy Walk", and our own more naive method, "Greedy Policy Walk" (GPW) designed to work for the large state space of Chess.

| ![Trace Plot of GPW Piece Weights on 1900-2000 ELO Player Moves](![trace_plot_GPW_1900_player_move_piece_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/8d6246a2-ff46-41b4-a4c3-30564cccfc5a)) | ![Trace Plot of GPW PST Weights on 1900-2000 ELO Player Moves](![trace_plot_GPW_1900_player_move_pst_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/e032689a-a006-4d5d-82df-83f4faed56a8)) |
|:------------------------------:|:------------------------------:|
| Title 1                        | Title 2                        |

| ![Trace Plot of GPW Piece Weights on Synthetic Engine Moves](![trace_plot_GPW_1900_sunfish_move_piece_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/0312216b-3df7-43dc-9fcc-5dd0f371a865)) | ![Trace Plot of GPW PST Weights on Synthetic Engine Moves](![trace_plot_GPW_1900_sunfish_move_pst_weights](https://github.com/gnf-bachelor/irl-chess/assets/98162688/470db9fe-f82d-4dec-9b61-e28392935e20)) |
|:------------------------------:|:------------------------------:|
| Title 3                        | Title 4                        |









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

First, setup your python environment. Make sure to use python 3.10 for all dependencies and submodules to work properly. You can either install the `conda` environment, [`irl_chess_env.yml`](irl_chess_env.yml), or you can make sure all the required packages are installed from `requirements.txt`. 

Then adjust the config files.
To select the desired model change the "model" parameter in the base_config.json in the experiment_config folder.
Each model has a folder containing its own config where parameters specific to the model are set. See the Config Description section.

When parameters in the base and model-configs are set to the desired values, simply run run_model.py
The results will be saved the results folder under a name that indicates what parameters where used for the run.

## Config Description



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

## Project structure




