# chess-irl
A project exploring approximate Inverse Reinforcement Learning (IRL) applications in Chess. 
This includes a modified version of the Bayesian Inverse Reinforcement Learning (BIRL) algortihm "Policy Walk", and our own more naive method, "Greedy Policy Walk" (GPW) designed to work for the large state space of Chess.

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

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## How to Run BIRL or GPW
For the Maia chess repository to work use python 3.10
The repository centers around the irl_chess source folder which contains the script run_model.py.
To select the desired model change the "model" parameter in the base_config.json in the experiment_config folder.
Each model has a folder containing its own config where parameters specific to the model are set.
When parameters in the base and model-configs are set to the desired values, simply run run_model.py
The results will be saved the results folder under a name that indicates what parameters where used for the run.
## How to build Maia chess components

After cloning the repository locally, don't forget to run the following commands in order to also clone the git submodules:

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




