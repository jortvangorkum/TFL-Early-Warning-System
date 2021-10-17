from pathlib import Path
import pandas
from pandas import DataFrame
import numpy
from sklearn.model_selection._split import train_test_split
from typing import Dict
from sklearn.naive_bayes import CategoricalNB


class NaiveBayes(object):

    def __init__(self):
        pass

    def run_model(self):
        """
        Train the model and report the accuracy on the test set.
        """
        x_train, x_test, y_train, y_test = self.get_data()
        model = CategoricalNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print("Accuracy: ", (y_test != y_pred).sum() / len(y_test))

    def get_data(self):
        """
        Retrieve the data and convert the one-hot encodings to single columns.
        Then split the data into train and test sets.
        """
        path = str(Path(__file__).parent.parent.joinpath("data", "data-normalization-output.csv"))
        df = pandas.read_csv(path)

        df.drop(df.columns[0], axis=1, inplace=True)
        avg_avg_score = df['average_score'].mean()
        df['average_score'] = df['average_score'].replace(numpy.nan, avg_avg_score)
        df.fillna(0, inplace=True)

        self.change_oh_cat("code_presentation", df)
        self.change_oh_cat("code_module", df)
        self.change_oh_cat("gender", df)
        self.change_oh_cat("region", df)
        self.change_oh_cat("highest_education", df)
        self.change_oh_cat("imd_band", df)
        self.change_oh_cat("age_band", df)
        self.change_oh_cat("disability", df)
        result_order = {'final_result__Distinction': 3, 'final_result__Fail': 0,
                        'final_result__Pass': 2, 'final_result__Withdrawn': 1}
        self.change_oh_cat("final_result", df, result_order)

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
