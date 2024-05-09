# chess_irl

A short description of the project.
USE PYTHON 3.10!! Otherwise maia-chess will not work
In order to use Maia chess engines clone this repository, then cd into it.
Next execute the following command which clones the 'maia-chess' repository into this one:

git clone https://github.com/gnf-bachelor/maia-chess.git

git submodule add https://github.com/gnf-bachelor/maia-chess.git maia_chess

Then go to https://lczero.org/play/download/ and download an appropriate backend. Place the contents in
a subfolder of maia-chess called 'lc0-exe-folder' (maia-chess/lc0-exe-folder). 

Rename 'maia-chess' to 'maia_chess' so it can be recognised by python as a module...

Run pip install -r maia_chess/requirements.txt

Now you can use maia_chess models using the following:

from maia_chess import load_maia_network

model = load_maia_network(1100, parent='maia_chess/')

This script keeps running in the background until stopped externally.

## Project structure

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
