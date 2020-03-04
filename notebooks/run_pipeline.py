import os
import pandas as pd
import seaborn as sns
from train_model import enrich_data_with_movies_features, prepare_data_for_sklearn, train_model

TARGET_CUTOFF = 7.5
PROJECT_BASE_DIR = "/home/rohail/projects/imdb_ratings/"
model_save_dir = "models/"
data_dir = "data/raw/"
plot_write_dir = "reports/figures"
idx_columns = ["imdb_title_id", "title", "original_title"]
main_fname = "movies.csv"

# read in data
df = pd.read_csv(os.path.join(PROJECT_BASE_DIR, data_dir, main_fname))

# add features
df_subset = enrich_data_with_movies_features(df, idx_columns, target_columns = ["avg_vote"], dummy = False)
# remove rows where we are missing user/critic reviews
missing_user_critic_review_mask = (
    df[["reviews_from_users", "reviews_from_critics"]].isna().any("columns")
)
df_subset = df_subset.loc[~missing_user_critic_review_mask].copy()


# scaling data
df_model, _ = prepare_data_for_sklearn(
    df_subset, dummy_encoding=True, target_variable="avg_vote", idx_columns=idx_columns
)

# binarize target
df_model.loc[:, "avg_vote_flag"] = (df_model.avg_vote >= TARGET_CUTOFF).astype(int)
class_balance = df_model.avg_vote_flag.value_counts()
percentage_of_total = float(
    (class_balance[class_balance.index == 1]) / class_balance.sum()
)
percent = "{:.2%}".format(percentage_of_total)

if class_balance.loc[1] < 5:
    print("Not fitting model due to lack of positive samples.")

print(
    f"The positive class represents {percent} of the total instances.\n"
    f"Positives: {class_balance.loc[1]}, negatives: {class_balance.loc[0]}"
)
target_columns = ["avg_vote", "avg_vote_flag"]


# train model and dump to disk

# regression
parameters = {
    "plot_write_dir": os.path.join(PROJECT_BASE_DIR, plot_write_dir),
    "model_save_dir": os.path.join(PROJECT_BASE_DIR, model_save_dir),
    "model_type": "regression",  # , classification
    "idx_columns": idx_columns,
    "test_set_size": 0.1,
    "training_parameters": {
        "class_weight": "balanced",  # vs providing sample weight to fit --> does it make a difference?
        "n_jobs": -1,
        "max_iter" : 1000,
        "scoring": "balanced_accuracy",
    },
}

reg_model, df_reg_coefs, reg_x_test, reg_y_test = train_model(
    df_model,
    parameters=parameters, 
)

# classification
parameters.update({"model_type": "classification"})
clf_model, df_clf_coefs, clf_x_test, clf_y_test = train_model(
    df_model,
    parameters=parameters,
)
assert clf_x_test.equals(reg_x_test)
y_test = clf_y_test.join(reg_y_test.drop(columns=idx_columns))
assert len(y_test) == len(reg_x_test)
assert all(y_test.isna().sum() == 0)
assert all(y_test.index == reg_x_test.index)

# write to disk

df_model.to_csv(os.path.join(PROJECT_BASE_DIR, model_save_dir, "df_model.csv"), index = False)
clf_x_test.to_csv(os.path.join(PROJECT_BASE_DIR, model_save_dir, "x_test.csv"), index = False)
y_test.to_csv(os.path.join(PROJECT_BASE_DIR, model_save_dir, "y_test.csv"), index = False)