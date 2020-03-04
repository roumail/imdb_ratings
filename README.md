imdb_ratings
==============================

Objective
-----------------

Build a predictive model for predicting whether or not a movie will achieve an average voting
score > 7.5.


Introduction
-----------------
This repository, available on [github](https://github.com/roumail/imdb_ratings), is the beginning of a predictive modeling project. The main work has been performed in the notebook directory. The src/ directory contains some boiler plate code so that we'd be able to run the data pipeline using commands such as make date, etc. 

Let's consider the main directory, notebooks, in more detail.

Methodology
------------------

There are three main notebooks, to be followed in order. The first notebook is the main one, where we build an end to end model for training and validating our predictive models without jumping into feature engineering. Feature engineering is addressed in the second notebook but was not prioritized too much for the purposes of this illustration since it is a rather long process (but one that leads to the most gains!). The third notebook builds on the first two notebooks, using train_model.py for imports, making a model that uses features, mostly coming from the movies data set.

__Feature importance__:
There are two main approaches I would consider here - the one I take currently simply looks at the coefficients from the fitted model to assess how important they are. Using a tree based model would lead to feature importance plots out of the box. Yet another approach would be to use blackbox methods such as SHAP.


Modelling
--------------

This problem could have been approached as a regression or classification problem. As seen in the first notebook, I used both and compared their performances to choose the best approach. 


Project Organization
------------

    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering)
    │                         and a short `-` delimited description, e.g.
    │                         `1.0-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
