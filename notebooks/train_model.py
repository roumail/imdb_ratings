import os
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Set
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from yellowbrick.classifier import DiscriminationThreshold
from yellowbrick.regressor import ResidualsPlot
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, balanced_accuracy_score
import time, datetime
import numpy as np
import joblib
import warnings


def summarise_classifier_performance(
    model: LogisticRegression,
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> plt.Figure:
    """
    Summarize model performance on train and test set and ROC on test set

    Args:
        model: A trained LogisticRegressionCV object
        x_train: Train subset for features
        y_train: Train subset for target variable
        x_test: Test subset for features
        y_test: Test subset for target variable

    Returns:
        A matplotlib object visualizing ROC curve on test set
    """
    if isinstance(model, LogisticRegressionCV):
        chosen_C = model.C_[0]
        print(f"Optimal C parameter: {chosen_C}")
    elif isinstance(model, LogisticRegression):
        chosen_C = model.get_params().get("C")
        print(f"C parameter: {chosen_C}")

    y_train_scores = model.decision_function(x_train)
    y_test_scores = model.decision_function(x_test)
    auc_train = roc_auc_score(y_train, y_train_scores).round(3)
    auc_test = roc_auc_score(y_test, y_test_scores).round(3)
    print(f"AUC score on train set: {auc_train}")
    print(f"AUC score on test set : {auc_test}")
    fpr, tpr, auc_thresholds = roc_curve(y_test, model.predict_proba(x_test)[:, 1])
    roc_plt = plot_roc_curve(
        fpr, tpr, label=f"AUC score on test set: {auc_test} for chosen C: {chosen_C}"
    )

    return roc_plt


def plot_roc_curve(fpr: np.array, tpr: np.array, label=None) -> plt.Figure:
    """
    The ROC curve, modified from
    Hands-On Machine learning with Scikit-Learn and TensorFlow; p.91

    Args:
        fpr: Increasing false positive rates such that element i is the false
        positive rate of predictions with score >= thresholds[i]
        tpr: Increasing true positive rates such that element i is the true
        positive rate of predictions with score >= thresholds[i].
        label: Title for the plot

    Returns:
        A matplotlib object visualizing ROC curve for provided tpr and fpr
    """
    roc_fig = plt.figure(figsize=(8, 8))
    plt.title("ROC Curve" if not label else label)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([-0.005, 1, 0, 1.005])
    plt.xticks(np.arange(0, 1, 0.05), rotation=90)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    return roc_fig


def get_model_coefficients(
    model: Union[LogisticRegression, LinearRegression], features: pd.DataFrame
) -> pd.DataFrame:
    """
    Retrieve model coefficients

    Args:
        model: A trained LogisticRegression
        features: The pd.DataFrame containing features

    Returns:
        A pd.DataFrame containing the coefficients
    """
    intercept = model.intercept_
    intercept = (
        np.array(intercept).reshape((1,)) if isinstance(intercept, float) else intercept
    )
    df_coefs = pd.DataFrame(
        {"coefficients": np.concatenate((intercept, model.coef_.flatten()))},
        index=["intercept"] + features.columns.tolist(),
    )
    if isinstance(model, LogisticRegression):
        df_coefs.loc[:, "coeff_odds_exponentiated"] = np.exp(df_coefs["coefficients"])

    return df_coefs.reset_index()


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    parameters: Dict,
    load_from_disk: Optional[str] = None,
) -> Tuple[LogisticRegression, pd.DataFrame]:
    """
    Helper to train the model or load from disk
    """
    model_type = parameters["model_type"]
    test_set_size = parameters["test_set_size"]
    model_save_dir = parameters["model_save_dir"]
    idx_columns = parameters["idx_columns"]

    plot_write_dir = parameters.get("plot_write_dir") 

    if model_type == "regression":
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_set_size, stratify=X["avg_vote_flag"], random_state=123
        )
        cv = ShuffleSplit(n_splits=5, random_state=0)
        cv.get_n_splits(x_train, y_train)
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_set_size, stratify=y, random_state=123
        )

        cv = StratifiedShuffleSplit(n_splits=5, random_state=0)
        cv.get_n_splits(x_train, y_train)

    # Create lookup between index and train to retrieve idx columns for y_test ....
    idx2lookup_cols = x_test.loc[:, idx_columns].copy()
    # drop the extra target column... leaking information...
    target_column = [col for col in x_train.columns if "avg_vote" in col]
    x_train = x_train.drop(columns=idx_columns + target_column)
    x_test = x_test.drop(columns=idx_columns + target_column)

    if load_from_disk:
        path2model = os.path.join(model_save_dir, load_from_disk)
        assert os.path.exists(path2model)
        print(f"Loading model from {path2model}")
        model = joblib.load(path2model)
        df_coefficients = get_model_coefficients(model, x_train)
        y_test = idx2lookup_cols.join(y_test)
        assert all(y_test.isna().sum() == 0)
        return model, df_coefficients, x_test, y_test

    start = time.time()
    if model_type == "classification":
        fit_params = parameters["training_parameters"]
        model = LogisticRegression(verbose=0, penalty="l2", random_state=123)
        # Select optimum threshold
        # Queue rate: This metric describes the percentage of instances that must be reviewed. If review has a high cost
        # (e.g. fraud prevention) then this must be minimized with respect to business requirements; if it doesn’t
        # (e.g. spam filter), this could be optimized to ensure the inbox stays clean.
        # we want this rate to be low, since we don't want to miss high rated target even if we create false positives..
        threshold_fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            visualizer = DiscriminationThreshold(model, ax=ax, is_fitted=False, cv=cv)
            visualizer.fit(x_train, y_train, n_trials=1, **fit_params)
            visualizer.finalize()
            threshold_fig.savefig(
                os.path.join(plot_write_dir, f"{model_type}_threshold_fig.png")
            )

        roc_fig = summarise_classifier_performance(
            model, x_train, y_train, x_test, y_test
        )
        roc_fig.savefig(os.path.join(plot_write_dir, "roc_fig.png"))
    else:
        regression_diags_fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        model = LinearRegression()
        visualizer = ResidualsPlot(model)
        visualizer.fit(x_train, y_train)
        visualizer.finalize()
        regression_diags_fig.savefig(
            os.path.join(plot_write_dir, f"{model_type}_residual_diagnostics_fig.png")
        )
        visualizer.score(x_test, y_test)  # Evaluate the model on the test data

    end = time.time()
    print(f"The training took {np.round((end - start), 3)} seconds", "\n")
    value = datetime.datetime.fromtimestamp(end)
    model_path = (
        f"{model_save_dir}/{model_type}_{value.strftime('%Y_%m_%d_%H_%M')}.joblib"
    )
    joblib.dump(model, model_path)

    # retrieve coefficients
    df_coefficients = get_model_coefficients(model, x_train)
    y_test = idx2lookup_cols.join(y_test)
    assert all(y_test.isna().sum() == 0)

    return (
        model,
        df_coefficients,
        x_test,
        y_test,
    )


def train_model(
    df_model: pd.DataFrame, parameters: Dict, load_from_disk: Optional[str] = None
) -> Tuple[LogisticRegression, pd.DataFrame]:
    """
    This function fits either a LogisticRegression or LinearRegression model from sklearn and returns the trained model
    and fitted coefficients to be visualized in a feature importance plot

    Args:
        df_model: A dataframe, which has already passed through scaling/one hot encoding, etc to be used for training
        the model. The target variable is collected from the parameters dictionary
        parameters: Contains training parameters as well as others needed for model fitting

    Returns:
        A List containing:
            1. Fitted LogisticRegressionCV object
            2. pd.DataFrame containing the fitted coefficients

    """
    # collect parameters
    model_type = parameters["model_type"]
    assert model_type in ["regression", "classification"]
    if load_from_disk:
        assert model_type in load_from_disk
    target = "avg_vote_flag" if "classification" in model_type else "avg_vote"

    nl = "\n"
    print(
        f"Diagnostic plots for this model can be found in the following directory: {nl + plot_write_dir}",
        f"\nThe model itself is saved in the following directory: {model_save_dir}",
        "\n",
        sep="",
    )
    X = df_model.drop(columns=target).copy()
    y = df_model[target].copy()

    # fit model
    model, df_coefficients, x_test, y_test = fit_model(
        X, y, parameters, load_from_disk=load_from_disk
    )

    return model, df_coefficients, x_test, y_test


def predict_models_on_data(
    reg_model: LinearRegression,
    clf_model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    features: List[str],
    parameters: Dict,
):
    classification_threshold = parameters["classification_threshold"]
    regression_threshold = parameters["regression_threshold"]
    x_test = x_test.loc[:, features].copy()
    prob_prediction = clf_model.predict_proba(x_test)[:, 1]
    classification_prediction = (prob_prediction > classification_threshold).astype(int)
    rating_prediction = reg_model.predict(x_test)
    x_test.loc[:, "clf_prob_prediction"] = prob_prediction
    x_test.loc[:, "reg_rating_prediction"] = rating_prediction
    regression_prediction = (rating_prediction >= regression_threshold).astype(int)
    x_test.loc[:, "regression_prediction"] = regression_prediction
    x_test.loc[:, "classification_prediction"] = classification_prediction
    print(x_test[["reg_rating_prediction", "clf_prob_prediction"]].describe())
    # merge together with y_text and return
    df_predict_test = x_test.join(y_test)
    assert all(df_predict_test.isna().sum() == 0)
    # check if predictions are the same
    print(
        "Regression and classification predictions the same?",
        df_predict_test["regression_prediction"].equals(
            df_predict_test["classification_prediction"]
        ),
    )
    print(
        "Balanced accuracy for regression: ",
        balanced_accuracy_score(
            df_predict_test["avg_vote_flag"], df_predict_test["regression_prediction"]
        ),
    )
    print(
        "Balanced accuracy for classification: ",
        balanced_accuracy_score(
            df_predict_test["avg_vote_flag"],
            df_predict_test["classification_prediction"],
        ),
    )
    return df_predict_test


def compare_methods(
    df_model: pd.DataFrame,
    reg_model: LinearRegression,
    clf_model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.DataFrame,
    train_movies_sample: Set,
    parameters: Dict,
):
    target_columns = parameters["target_columns"]
    idx_columns = parameters["idx_columns"]
    features = list(set(df_model.columns) - set(target_columns).union(idx_columns))

    # make predictions on test data
    print("Making predictions on test data")
    df_predict_test = predict_models_on_data(
        reg_model, clf_model, clf_x_test, y_test, features, parameters
    )
    
    # make predictions on train data -- movies I know where we can identify gaps...
    x_train = df_model.loc[df_model.title.isin(train_movies_sample), features].copy()
    y_train = df_model.loc[
        df_model.title.isin(train_movies_sample), y_test.columns
    ].copy()
    print("Making predictions on sample data from train data")
    df_predict_train = predict_models_on_data(
        reg_model, clf_model, x_train, y_train, features, parameters
    )

    return df_predict_test, df_predict_train
