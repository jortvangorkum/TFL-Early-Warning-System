import pandas
from typing import Dict, List
from pathlib import Path
from pandas import DataFrame
from sklearn.model_selection._split import train_test_split
from classification.decision_tree_classifier import TreeClassifier
import matplotlib.pyplot as plt

DATA_PATH = Path(__file__).parent.parent.joinpath("data")


def create_trees():
    target_names = ["Fail", "Pass"]
    # target_names = ["Fail", "Withdrawn", "Pass", "dist"]
    trees = list()
    for i in range(7):
        print("making tree", i+1)
        x_train, x_test, y_train, y_test = prep_tree_data(i + 1)
        tree = TreeClassifier(x_train, x_test, y_train, y_test, target_names)
        tree.run_model()
        trees.append(tree)

    return trees


def plot_scores(trees: List[TreeClassifier]):
    """
    Plots the train accuracy, test accuracy, test precision and test recall, and saves them in the
    data folder.
    """
    scores = DataFrame(columns=["Train Accuracy", "Test Accuracy", "Precision", "Recall"])
    for i, tree in enumerate(trees):
        train_accuracy, test_accuracy, test_precision, test_recall = tree.get_scores()
        scores.loc[i+1] = [train_accuracy, test_accuracy, test_precision, test_recall]

    for i, col in enumerate(scores.columns.values):
        colour = plt.get_cmap("Set1")(i)
        ax = scores.plot(kind='bar', y=col, color=colour)
        ax.set_ylim([0, 1])
        plt.xticks(rotation=None)
        plt.title(col)
        plt.xlabel("Months of assessments")
        plt.ylabel(col)
        ax.get_legend().remove()

        path = str(DATA_PATH.joinpath(col + ".png"))
        plt.savefig(path)


def prep_tree_data(number: int):
    """
    Retrieve the data and convert the one-hot encodings to single columns.
    Then split the data into train and test sets.
    """
    filename = "data-before-normalization-{}-out-of-7.csv".format(number)
    path = str(DATA_PATH.joinpath("data-splitted", filename))
    df = pandas.read_csv(path)

    df.drop(df.columns[0], axis=1, inplace=True)
#     avg_avg_score = df['average_score'].mean()
#     df['average_score'].replace(numpy.nan, avg_avg_score, inplace=True)
    assessments = [x for x in df.columns.values if x.split("_")[0] == "assessment"]
    df['average_score'] = df[assessments].mean(skipna=True, axis=1)
    for assessment in assessments:  # somehow he doesn't want to fillna in a batch?
        df[assessment].fillna(df['average_score'], inplace=True)
    df.dropna()

    change_oh_cat("gender", df)
    change_oh_cat("highest_education", df)
    change_oh_cat("imd_band", df)
    change_oh_cat("age_band", df)
    change_oh_cat("disability", df)
    result_order = {'final_result__Fail': 0,  'final_result__Withdrawn': 2,
                    'final_result__Pass': 1, 'final_result__Distinction': 3}
    change_oh_cat("final_result", df, result_order)
    df["final_result"].replace(2, 0, inplace=True)
    df["final_result"].replace(3, 1, inplace=True)

    target = df["final_result"]
    df.drop(["final_result"], axis=1, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.1,
                                                        random_state=32, shuffle=True,
                                                        stratify=target)

    return x_train, x_test, y_train, y_test


def change_oh_cat(target: str, df: DataFrame, pref_order: Dict[int, int] = None):
    """
    Change the one-hot coding to a single column with numbers for the different categories.
    Optionally, a given order for the categories can be given in pref_order.
    """
    target_cols = [x for x in df.columns.values if x.split("__")[0] == target]

    if pref_order is None:
        pref_order = {x: i for i, x in enumerate(target_cols)}

    str_row = df[target_cols].idxmax(1)
    label_row = [pref_order[x] for x in str_row]
    df[target] = label_row
    df.drop(target_cols, axis=1, inplace=True)
