import os
import pandas
from typing import Dict, List
from pathlib import Path
from pandas import DataFrame
from sklearn.model_selection._split import train_test_split
from classification.decision_tree_classifier import TreeClassifier
import matplotlib.pyplot as plt
from _collections import defaultdict
from classification.tree_explainer import TreeExplainer

import seaborn as sns
sns.set_theme()

DATA_PATH = Path(__file__).parent.parent.joinpath("data")


class TreeManager(object):

    def __init__(self):
        self.trees = None

    def create_trees(self, importance_values: List[int]) -> None:
        """
        Make the decision trees and train them. The trees are stored in this class, separated by
        the importance of the fail class.
        """
        target_names = ["Fail", "Pass"]
        trees = defaultdict(list)
        for importance in importance_values:
            for i in range(7):
                print(f'making tree for week {i + 1} with importance {importance}')
                x_train, x_test, y_train, y_test = self.prep_tree_data(i + 1)
                tree = TreeClassifier(x_train, x_test, y_train, y_test, target_names, importance)
                tree.run_model()
                trees[importance].append(tree)

        self.trees = trees

    def explain_sample(self, importance: float, months: int, index: int) -> None:
        """
        Explain a sample from the test set.
        :param importance: importance of the Fail class. Used as a multiplier for its class weight
        :param months: how many months of assessments are taken into account
        :param index: the index of the sample in the test set
        """
        tree = self.trees[importance][months]
        explainer = TreeExplainer(tree.model, tree.feature_names, tree.target_names)
        explainer.print_path_and_foil(tree.x_test.iloc[[index]])

    def plot_scores(self):
        """
        Plots the train accuracy, test accuracy, test precision and test recall, and saves them in
        the data folder.
        Also saves the scores in the data folder in 'scores.csv'
        """
        results_path = DATA_PATH.joinpath("results")
        if not results_path.is_dir():
            os.mkdir(results_path)

        scores = DataFrame(columns=["Import", "Months", "Train Accuracy", "Test Accuracy",
                                    "Precision", "Recall"])

        for importance, tree_list in self.trees.items():
            for i, tree in enumerate(tree_list):
                train_accuracy, test_accuracy, precision, recall = tree.get_scores()
                scores.loc[len(scores)] = [importance, i+1, train_accuracy, test_accuracy,
                                           precision, recall]

        for i, col in enumerate(scores.columns.values):
            if col == "Months" or col == "Import":
                continue

            pivot_scores = pandas.pivot_table(scores, values=col, index="Months", columns="Import")
            ax = pivot_scores.plot(kind='bar')
            ax.set_ylim([0, 1])
            plt.xticks(rotation=None)
            plt.title(col)
            plt.xlabel("Months of assessments")
            plt.ylabel(col)
            plt.legend(loc="lower right")

            path = str(results_path.joinpath(col + ".png"))
            plt.savefig(path)

        score_path = str(results_path.joinpath("scores.csv"))
        scores.to_csv(score_path)


    def plot_weeks(self):

        def annotate_text(dataframe, column_y):
            for x,y in zip(dataframe['Month'], dataframe[column_y]):
                label = "{:.2f}".format(y)
                plt.annotate(label, (x,y), textcoords="offset points", xytext=(0,10), ha='center')

        for importance, tree_list in self.trees.items():
            scores = DataFrame(columns=["Importance", "Month", "Accuracy", "Precision", "Recall"])
            # loop over trees
            for i, tree in enumerate(tree_list):
                train_accuracy, test_accuracy, precision, recall = tree.get_scores()
                scores.loc[len(scores)] = [importance, i+1, test_accuracy, precision, recall]
                
            # plot lines 
            with sns.color_palette("Set2"):
                plt.plot('Month', 'Accuracy', data=scores, label='Accuracy')
                plt.plot('Month', 'Precision', data=scores, label='Precision')
                plt.plot('Month', 'Recall', data=scores, label='Recall')

            plt.ylabel('Value')
            plt.xlabel('Month')
            plt.legend()
            plt.savefig(f'data/results/result-tree-importance-{importance}-per-month.pdf', dpi=300)
            plt.savefig(f'data/results/result-tree-importance-{importance}-per-month.svg', dpi=300)
            plt.savefig(f'data/results/result-tree-importance-{importance}-per-month.png', dpi=300)
            plt.clf()
            # print(scores)


    def prep_tree_data(self, number: int):
        """
        Retrieve the data and convert the one-hot encodings to single columns.
        Then split the data into train and test sets.
        """
        filename = "data-before-normalization-{}-out-of-7.csv".format(number)
        path = str(DATA_PATH.joinpath("data-splitted", filename))
        df = pandas.read_csv(path)

        df.drop(df.columns[0], axis=1, inplace=True)
        assessments = [x for x in df.columns.values if x.split("_")[0] == "assessment"]
        df['average_score'] = df[assessments].mean(skipna=True, axis=1)
        for assessment in assessments:  # somehow he doesn't want to fillna in a batch?
            df[assessment].fillna(df['average_score'], inplace=True)
        clicks = [x for x in df.columns.values if x.split("_")[0] == "vle"]
        df['vle_click_average'] = df[clicks].mean(skipna=True, axis=1)
        for click in clicks:  # somehow he doesn't want to fillna in a batch?
            df[click].fillna(df['vle_click_average'], inplace=True)
        df.dropna()

        self.change_oh_cat("gender", df)
        self.change_oh_cat("highest_education", df)
        self.change_oh_cat("imd_band", df)
        self.change_oh_cat("age_band", df)
        self.change_oh_cat("disability", df)
        result_order = {'final_result__Fail': 0,  'final_result__Withdrawn': 2,
                        'final_result__Pass': 1, 'final_result__Distinction': 3}
        self.change_oh_cat("final_result", df, result_order)
        df["final_result"].replace(2, 0, inplace=True)
        df["final_result"].replace(3, 1, inplace=True)

        target = df["final_result"]
        df.drop(["final_result"], axis=1, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.1,
                                                            random_state=32, shuffle=True,
                                                            stratify=target)

        return x_train, x_test, y_train, y_test

    def change_oh_cat(self, target: str, df: DataFrame, pref_order: Dict[int, int] = None):
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
