from __future__ import annotations
from sklearn.tree._classes import DecisionTreeClassifier
from pandas.core.frame import DataFrame
from _collections import defaultdict
from typing import Tuple, Dict, List
import itertools
from classification.decision import Decision, Sign


class TreeExplainer(object):
    """
    Explains a decision tree.
    """

    def __init__(self, model: DecisionTreeClassifier, feature_names, target_names):
        self.model = model
        self.feature_names = feature_names
        self.target_names = target_names

    def _convert_to_decisions(self, node_ids: List[int], leaf: int, sample: DataFrame = None) \
            -> List[Decision]:
        """
        Convert the list of nodes to a list of decisions.
        :param node_ids: id's of the nodes in the tree. Are sorted leaf to root
        :param leaf: the id of the leaf to which this path leads
        :param sample: sample to take a value from. Can be left out for foils
        """
        tree = self.model.tree_
        decisions = list()

        for i, node_id in reversed(list(enumerate(node_ids))):
            child = node_ids[i-1] if i > 0 else leaf
            sign = Sign.seq if child == tree.children_left[node_id] else Sign.larger
            feature = self.feature_names[tree.feature[node_id]]
            value = sample.iloc[0][feature] if sample is not None else None
            decision = Decision(feature, tree.feature[node_id], sign,
                                tree.threshold[node_id], sample is not None, value)
            decisions.append(decision)

        return decisions

    def _aggregate_features(self, node_ids: List[int], leaf: int, sample: DataFrame = None) \
            -> List[Decision]:
        """
        Gets a list of node ids and sorts them according to the feature on which they split.
        Then filter out nodes that test the same value with the same sign (< or >), and keep
        the most specific one. If the same feature is tested with both > and <, a 'between' node
        is created.
        :return: all decisions, not sorted
        """
        features_sample = defaultdict(list)
        decisions = self._convert_to_decisions(node_ids, leaf, sample)
        for decision in decisions:
            features_sample[decision.feature_id].append(decision)

        for nodes in features_sample.values():
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

        list_decisions = []
        for nodes in features_sample.values():
            list_decisions.extend(nodes)

        return list_decisions

    def _print_comparison(self, decisions_sample: List[Decision], decisions_foil: List[Decision],
                          sample: DataFrame) -> None:
        """
        Prints the results of the comparison.
        """
        print("Contrastive rules:")
        for decision in decisions_foil:
            verdict = decision.str_apply(sample.iloc[0][decision.feature_name])
            if verdict is None:
                continue
            print(verdict)

        if decisions_sample:
            print("")
            print("Additional rules used to come to the verdict of the sample:")
            for decision in decisions_sample:
                print(decision)

    def _inspect_tree(self) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
        """
        Go through the tree to get the parents of the leaves.

        :return:
        - Dict[leaf_id, List[node_ids]] leaf to parents. Parents are sorted leaf to root
        - Dict[target_id, List[leaf_ids]] target id (majority class) to corresponding leaves
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

        return parents, value_leaves

    def report_foils(self, sample: DataFrame, parents: Dict[int, List[int]] = None,
                     value_leaves: Dict[int, List[int]] = None) -> None:
        """
        Report for each class to which the sample wasn't assigned, what the difference is between
        its node and the closest node with a different class.

        TODO: make it so that rules of the same feature are compared between the
        foil and the sample.
        """
        if (parents is None) or (value_leaves is None):
            parents, value_leaves = self._inspect_tree()
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

            print("Closest leaf resulting in a {}: {}, distance = {}.".
                  format(self.target_names[value], close_leaf, close_dist + 1))

            sample_cut_index = sample_parents.index(close_parent)
            sample_lowers = [*sample_parents[:sample_cut_index], close_parent]
            leaf_cut_index = parents[close_leaf].index(close_parent)
            leaf_lowers = [*parents[close_leaf][:leaf_cut_index], close_parent]

            decisions_sample = self._aggregate_features(sample_lowers, sample_leaf, sample)
            decisions_foil = self._aggregate_features(leaf_lowers, close_leaf)
            decisions_sample = [x for x in decisions_sample if x not in decisions_foil]

            self._print_comparison(decisions_sample, decisions_foil, sample)

    def print_path(self, sample: DataFrame, parents: Dict[int, List[int]] = None):
        """
        Prints the complete path that led to the classification of the sample.
        """
        tree = self.model.tree_
        if parents is None:
            parents, _ = self._inspect_tree()
        leaf = self.model.apply(sample)[0]
        node_ids = parents[leaf]
        predicted = self.model.predict(sample)[0]

        print("Rules that predict sample {}:".format(sample.index[0]))
        decisions = self._convert_to_decisions(node_ids, leaf, sample)
        for node_id, decision in zip(node_ids, reversed(decisions)):
            print("Node {}: {}".format(node_id, decision))

        print("Therefore, the expected result is:", self.target_names[predicted])
        print("The impurity at this leaf, {}, is: {}".format(leaf, tree.impurity[leaf]))
        print("Distribution at leaf: ", {k: v for k, v in
                                         zip(self.target_names, tree.value[leaf][0])})

    def print_path_and_foil(self, sample: DataFrame) -> None:
        """
        Give a full explanation: both the full path to the decision, and a contrastive explanation.
        """
        parents, value_leaves = self._inspect_tree()
        self.print_path(sample, parents)
        print("")
        self.report_foils(sample, parents, value_leaves)
