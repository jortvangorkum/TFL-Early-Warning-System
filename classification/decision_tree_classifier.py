from pathlib import Path
import pandas
from pandas import DataFrame
import numpy
from sklearn.model_selection._split import train_test_split
from sklearn.tree._classes import DecisionTreeClassifier
from classification.tree_explainer import TreeExplainer
from typing import Dict
from sklearn.metrics._classification import precision_score, recall_score


class TreeClassifier(object):

    def __init__(self):
        pass

    def run_model(self):
        """
        To check: random forest (NAH)
        """
        self.x_train, self.x_test, self.y_train, self.y_test, self.target_names = self.get_data()
        self.feature_names = self.x_train.columns.values

        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, random_state=43,
                                            class_weight="balanced")
        self.model.fit(self.x_train, self.y_train)

        explainer = TreeExplainer(self.model, self.feature_names, self.target_names)
        explainer.report_foils(self.x_train.head(1))
        explainer.get_path(self.x_train.head(1))

    def print_scores(self) -> None:
        """
        Compute scores and print them.
        """
        train_accuracy = self.model.score(self.x_train, self.y_train)
        test_accuracy = self.model.score(self.x_test, self.y_test)

        y_test_pred = self.model.predict(self.x_test)
        test_precision = precision_score(y_test_pred, self.y_test)
        test_recall = recall_score(y_test_pred, self.y_test)

        print("train accuracy:", train_accuracy)
        print("test accuracy:", test_accuracy)
        print("test precision:", test_precision)
        print("test recall:", test_recall)

    def get_data(self):
        """
        Retrieve the data and convert the one-hot encodings to single columns.
        Then split the data into train and test sets.
        """
        path = str(Path(__file__).parent.parent.joinpath("data", "data-before-normalization-output.csv"))
        df = pandas.read_csv(path)

        df.drop(df.columns[0], axis=1, inplace=True)
        avg_avg_score = df['average_score'].mean()
        df['average_score'].replace(numpy.nan, avg_avg_score, inplace=True)
        assessments = [x for x in df.columns.values if x.split("_")[0] == "assessment"]
        for assessment in assessments:  # somehow he doesn't want to fillna in a batch?
            df[assessment].fillna(df['average_score'], inplace=True)
        df.dropna()

        self.change_oh_cat("gender", df)
        # self.change_oh_cat("region", df)  # leaving region as one-hot bc it's not ordered
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
        target_names = ["Fail", "Pass"]
        # target_names = ["Fail", "Withdrawn", "Pass", "dist"]

        x_train, x_test, y_train, y_test = train_test_split(df, target, test_size=0.1,
                                                            random_state=32, shuffle=True,
                                                            stratify=target)

        return x_train, x_test, y_train, y_test, target_names

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
