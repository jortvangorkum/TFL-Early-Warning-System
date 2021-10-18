from pathlib import Path
import pandas
from pandas import DataFrame
import numpy
from sklearn.model_selection._split import train_test_split
from sklearn.tree._classes import DecisionTreeClassifier
from _collections import defaultdict
from typing import Dict, Tuple, List


class TreeClassifier(object):

    def __init__(self):
        pass

    def run_model(self):
        x_train, x_test, y_train, y_test, target_names = self.get_data()
        self.feature_names = x_train.columns.values
        self.target_names = target_names

        self.model = DecisionTreeClassifier(max_depth=7, min_samples_split=50, random_state=43)
        self.model.fit(x_train, y_train)
        train_score = self.model.score(x_train, y_train)
        test_score = self.model.score(x_test, y_test)
        print("train accuracy:", train_score)
        print("test accuracy:", test_score)
        self.get_path(x_train.head(1))
        self.report_foils(x_train.head(1))

    def report_foils(self, sample: DataFrame):
        """
        Report for each class to which the sample wasn't assigned, what the difference is between
        its node and the closest node with a different class.

        TODO: make it so that rules of the same feature are aggregated, and compared between the
        foil and the sample.
        """
        tree = self.model.tree_
        parents, value_leaves = self.inspect_tree()
        sample_class = self.model.predict(sample)[0]
        sample_leaf = self.model.apply(sample)[0]
        sample_parents = parents[sample_leaf]

        for value, leaves in value_leaves.items():
            if value == sample_class:
                continue

            close_leaf = None
            close_dist = float('inf')
            close_parent = None
            for leaf in leaves:
                leaf_parents = parents[leaf]
                overlap = set(sample_parents).intersection(leaf_parents)
                if not overlap:
                    continue

                closest_overlap = (None, float('inf'))
                for shared_p in overlap:
                    distance = sample_parents.index(shared_p) + leaf_parents.index(shared_p)
                    if distance < closest_overlap[1]:
                        closest_overlap = (shared_p, distance)
                if closest_overlap[1] < close_dist:
                    close_leaf = leaf
                    close_dist = closest_overlap[1]
                    close_parent = closest_overlap[0]

            print("Closest leaf in class {}: {}, distance = {}.".format(self.target_names[value],
                                                                        close_leaf, close_dist))

            sample_cut_index = sample_parents.index(close_parent)
            sample_lowers = [*sample_parents[:sample_cut_index], close_parent]
            leaf_cut_index = parents[close_leaf].index(close_parent)
            leaf_lowers = [*parents[close_leaf][:leaf_cut_index], close_parent]

            print("Features of the sample:")
            for i, node in reversed(list(enumerate(sample_lowers))):
                feature = tree.feature[node]
                child = sample_lowers[i-1] if i > 0 else sample_leaf
                sign = "smaller" if child == tree.children_left[node] else "larger"
                threshold = tree.threshold[node]
                if node in leaf_lowers:
                    print("The {} of the sample is {} than {}, while foil's is not.".format(
                        self.feature_names[feature], sign, threshold))
                else:
                    print("The {} of the sample is {} than {}.".format(
                        self.feature_names[feature], sign, threshold))

            print("Features of the foil leaf:")
            for i, node in reversed(list(enumerate(leaf_lowers))):
                feature = tree.feature[node]
                child = leaf_lowers[i-1] if i > 0 else close_leaf
                sign = "smaller" if child == tree.children_left[node] else "larger"
                threshold = tree.threshold[node]
                print("The {} of the leaf is {} than {}.".format(
                        self.feature_names[feature], sign, threshold))

    def inspect_tree(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Go through the tree to get the parents of the leaves.

        :return:
        - Dict[leaf_id, List[node_ids] leaf to parents
        - Dict[target_id, List[leaf_ids] target id (majority class) to corresponding leaves
        """
        tree = self.model.tree_
        parents = dict()
        value_leaves = defaultdict(list)
        leaves = list()

        def recurse(node: int, depth: int, parent: int):
            parents[node] = [parent, *parents.get(parent, [])]
            if tree.children_left[node] != tree.children_right[node]:
                recurse(tree.children_left[node], depth + 1, node)
                recurse(tree.children_right[node], depth + 1, node)
            else:
                val = tree.value[node][0].argmax()
                value_leaves[val].append(node)
                leaves.append(node)

        recurse(0, 0, None)

        parents = {k: v[:-1] for k, v in parents.items() if k in leaves}
        # This prints all leaves with their parents
#         for value, leaves in value_leaves.items():
#             print("leaves with value", self.target_names[value])
#             for leaf in leaves:
#                 print(leaf)
#                 print("parents:", parents[leaf])

        return parents, value_leaves

    def get_path(self, sample: DataFrame):
        """
        Prints the path to the leaf of the sample.
        TODO: shorten by using the parent list from inspect_tree.
        """
        tree = self.model.tree_
        feature = tree.feature
        threshold = tree.threshold
        impurity = tree.impurity

        node_indicator = self.model.decision_path(sample)
        leaf_id = self.model.apply(sample)
        decision = self.model.predict(sample)

        sample_id = 0
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                            node_indicator.indptr[sample_id + 1]]

        print('Rules used to predict sample {id}:\n'.format(id=sample.index[sample_id]))
        for node_id in node_index:
            # continue to the next node if it is a leaf node
            if leaf_id[sample_id] == node_id:
                print("Therefore, the expected result is:", self.target_names[decision[sample_id]])
                print("The impurity at this leaf, {}, is: {}".format(node_id, impurity[node_id]))
                print("Distribution at leaf: ", tree.value[node_id])
                continue

            f_name = self.feature_names[feature[node_id]]
            # check if value of the split feature for sample 0 is below threshold
            if (sample.iloc[sample_id][f_name] <= threshold[node_id]):
                threshold_sign = "<="
            else:
                threshold_sign = ">"

            print("decision node {node} : {feature} = {value} {inequality} {threshold}".format(
                      node=node_id,
                      feature=f_name,
                      value=sample.iloc[sample_id][f_name],  # iloc[sample_id][f_name]
                      inequality=threshold_sign,
                      threshold=threshold[node_id]))

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
        result_order = {'final_result__Fail': 0,  'final_result__Withdrawn': 1,
                        'final_result__Pass': 2, 'final_result__Distinction': 3}
        self.change_oh_cat("final_result", df, result_order)

        target = df["final_result"]
        df.drop(["final_result"], axis=1, inplace=True)
        target_names = [x.split("__")[1] for x in result_order.keys()]

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
