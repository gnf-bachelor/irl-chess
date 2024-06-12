# chess-irl
A project exploring approximate Inverse Reinforcement Learning (IRL) applications in Chess. 
This includes a modified version of the Bayesian Inverse Reinforcement Learning (BIRL) algortihm "Policy Walk", and our own more naive method, "Greedy Policy Walk" (GPW) designed to work for the large state space of Chess.

## Project structure
The repository contains a modular setup for running the various algorithms with a single entry-point (run_model.py). All settings, including which model to use, are set in config files base_config.json and the model specific config.json.

![Pipeline](https://github.com/gnf-bachelor/irl-chess/assets/98162688/58f71093-a4ba-4f54-a9d4-a824788ff5c6)

The directory structure of the project looks like this:

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── chess_irl  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## How to Run BIRL or GPW

A short description of the project.
USE PYTHON 3.10!! Otherwise maia-chess will not work

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




