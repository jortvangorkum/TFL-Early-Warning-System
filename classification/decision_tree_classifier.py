from sklearn.tree._classes import DecisionTreeClassifier
from classification.tree_explainer import TreeExplainer
from sklearn.metrics._classification import precision_score, recall_score
from typing import Tuple


class TreeClassifier(object):

    def __init__(self, x_train, x_test, y_train, y_test, target_names):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_names = target_names

    def run_model(self):
        """
        To check: random forest (NAH)
        """
        self.feature_names = self.x_train.columns.values

        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, random_state=43,
                                            class_weight="balanced")
        self.model.fit(self.x_train, self.y_train)

        explainer = TreeExplainer(self.model, self.feature_names, self.target_names)
        explainer.report_foils(self.x_train.head(1))
        explainer.get_path(self.x_train.head(1))

    def get_scores(self) -> Tuple[float, float, float, float]:
        """
        Compute scores and return them.
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

        return train_accuracy, test_accuracy, test_precision, test_recall
