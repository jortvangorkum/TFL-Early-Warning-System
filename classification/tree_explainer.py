from __future__ import annotations
from sklearn.tree._classes import DecisionTreeClassifier
from pandas.core.frame import DataFrame
from _collections import defaultdict
from typing import Tuple, Dict, List
import itertools


class Decision(object):
    """
    Represents a decision node in the tree, optionally with the actual value that was tested
    in this node.
    TODO: add a 'between x and y' version when there are two rules who together form a range.
    """

    def __init__(self, feature_name: str, feature_id: int, sign: bool,
                 threshold: float, sample: bool, value: float = None):
        self.feature_name = feature_name
        self.feature_id = feature_id
        self.seq = sign
        self.threshold = threshold
        self.sample = sample
        self.value = value

    def __str__(self):
        if self.value is not None:
            return "The {} was {} than {} for the {}, namely {}.".\
                format(self.feature_name, self.seqstr(), round(self.threshold),
                       self.sampstr(), round(self.value))
        else:
            return "The {} was {} than {} for the {}.".\
                format(self.feature_name, self.seqstr(), round(self.threshold), self.sampstr())

    def aggregate(self, other: Decision) -> Decision:
        """
        Combine two decisions about the same feature if possible; use the most precise one.
        Assumes that it's about the same sample/foil and value.
        If combination is not possible, return None.
        """
        if other.feature_id == self.feature_id and other.seq == self.seq:
            if self.seq:
                new_threshold = min(self.threshold, other.threshold)
            else:
                new_threshold = max(self.threshold, other.threshold)
            return Decision(self.feature_name, self.feature_id, self.seq, new_threshold,
                            self.sample, self.value)
        else:
            return None

    def compare(self, other: Decision):
        """
        Compare the two decisions. If their rules are contrastive (one > other <) then a string
        is produced explaining the difference. Otherwise returns None.

        TODO: comparisons for other situations, discuss when to compare w group
        """
        if not self.sample:
            print("warning! compare called for foil instead of sample")

        if other.feature_id != self.feature_id:
            print("warning! compare called for other feature")
            return None
        else:
            if self.seq is not other.seq:
                if not other.apply(self.value):
                    return "The {} of the {} should be {} than {}, but it's not; it's {}.".\
                        format(self.feature_name, self.sampstr(), other.seqstr(),
                               round(other.threshold), round(self.value))
            else:
                return None

    def apply(self, value: float) -> bool:
        """
        Check whether the value passes the rule of this decision node.
        """
        return self.seq is (value <= self.threshold)

    def seqstr(self):
        return "smaller" if self.seq else "larger"

    def sampstr(self):
        return "sample" if self.sample else "foil"


class TreeExplainer(object):

    def __init__(self, model: DecisionTreeClassifier, feature_names, target_names):
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names

    def report_foils(self, sample: DataFrame):
        """
        Report for each class to which the sample wasn't assigned, what the difference is between
        its node and the closest node with a different class.

        TODO: make it so that rules of the same feature are aggregated, and compared between the
        foil and the sample.
        """
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

            features_sample = self.aggregate_features(sample_lowers, sample_leaf, sample)
            features_foil = self.aggregate_features(leaf_lowers, close_leaf)

            self.print_comparison(features_sample, features_foil, sample)

    def aggregate_features(self, node_ids: List[int], leaf: int, sample: DataFrame = None) \
            -> Tuple[Dict[int, List[Decision]], Dict[int, List[Decision]]]:
        """
        Gets a list of node ids and filters them according to the feature on which they split.
        Then filter out nodes that test the same value with the same sign (< or >), and keep
        the most specific one.
        """
        tree = self.model.tree_

        features_sample = defaultdict(list)
        for i, node_id in reversed(list(enumerate(node_ids))):
            child = node_ids[i-1] if i > 0 else leaf
            seq = child == tree.children_left[node_id]
            feature = self.feature_names[tree.feature[node_id]]
            value = sample.iloc[0][feature] if sample is not None else None
            decision = Decision(feature, tree.feature[node_id], seq, tree.threshold[node_id],
                                sample is not None, value)
            features_sample[tree.feature[node_id]].append(decision)

        for feature, nodes in features_sample.items():
            while len(nodes) > 1:
                combs = itertools.product(nodes, repeat=2)
                for x, y in combs:
                    if x == y:
                        continue
                    new_node = x.aggregate(y)
                    if new_node is not None:
                        nodes.remove(x)
                        nodes.remove(y)
                        nodes.append(new_node)
                        break
                else:
                    break

        return features_sample

    def print_comparison(self, features_sample: Dict[int, List[Decision]],
                         features_foil: Dict[int, List[Decision]], sample: DataFrame) -> None:
        """
        Prints the results of the comparison.
        """
        for i, feature in enumerate(self.feature_names):
            sample_only = list()
            foil_only = list()
            sample_nodes = features_sample.get(i, [])
            foil_nodes = features_foil.get(i, [])
            if foil_nodes is not None:
                foil_nodes = [x for x in foil_nodes if not x.apply(sample.iloc[0][feature])]

            if not sample_nodes and not foil_nodes:
                continue

            print("Comparison for feature", feature)
            sample_only.extend(sample_nodes)
            foil_only.extend(foil_nodes)
            if sample_nodes and foil_nodes:
                all_combs = itertools.product(sample_nodes, foil_nodes)
                compared = list()
                for s, f in all_combs:
                    comparison = s.compare(f)
                    if comparison is not None:
                        compared.extend([s, f])
                        print(comparison)
                sample_only = [x for x in sample_only if x not in compared]
                foil_only = [x for x in foil_only if x not in compared]

            if sample_only:
                print("Sample individual rules:")
                for d in sample_only:
                    print(d)
            if foil_only:
                print("Foil individual rules:")
                for d in foil_only:
                    print(d)
            print("")

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
                print("Distribution at leaf: ", tree.value[node_id][0])
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
