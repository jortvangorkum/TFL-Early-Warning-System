from sklearn.tree._classes import DecisionTreeClassifier
from sklearn.metrics._classification import precision_score, recall_score
from typing import Tuple
from sklearn.utils.class_weight import compute_class_weight


class TreeClassifier(object):

    def __init__(self, x_train, x_test, y_train, y_test, target_names, fail_import: float):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_names = target_names
        self.fail_import = fail_import

    def run_model(self):
        """
        To check: random forest (NAH)
        """
        self.feature_names = self.x_train.columns.values

        # adjust the class weight of the Fail class to prioritize
        class_weights = compute_class_weight(class_weight='balanced',
                                             classes=[i for i in range(len(self.target_names))],
                                             y=self.y_train)
        class_weights = {num: value for num, value in enumerate(class_weights)}
        class_weights[0] = class_weights[0] * self.fail_import

        self.model = DecisionTreeClassifier(max_depth=5, min_samples_split=50, random_state=43,
                                            class_weight=class_weights)
        self.model.fit(self.x_train, self.y_train)

        # ugh plot won't fit in the figure someone figure it out (hihi)
#         plt.figure(figsize=(30, 10))
#         skl_tree.plot_tree(self.model, fontsize=9, feature_names=self.feature_names,
#                            class_names=self.target_names, filled=True, label="none", precision=1,
#                            impurity=False)
#         plt.show()
#         plt.savefig("tree.png", dpi=100)

    def get_scores(self) -> Tuple[float, float, float, float]:
        """
        Compute scores and return them.
        """
        train_accuracy = self.model.score(self.x_train, self.y_train)
        test_accuracy = self.model.score(self.x_test, self.y_test)

        y_test_pred = self.model.predict(self.x_test)
        test_precision = precision_score(y_test_pred, self.y_test)
        test_recall = recall_score(y_test_pred, self.y_test)

        return train_accuracy, test_accuracy, test_precision, test_recall
