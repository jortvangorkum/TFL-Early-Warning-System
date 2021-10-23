from __future__ import annotations
from enum import Enum, auto
import numpy as np


class Sign(Enum):
    seq = auto()
    between = auto()
    larger = auto()


class Decision(object):
    """
    Represents a decision node in the tree, optionally with the actual value that was tested
    in this node.
    TODO: add a 'between x and y' version when there are two rules who together form a range.
    """

    def __init__(self, feature_name: str, feature_id: int, sign: Sign,
                 threshold: float, sample: bool, value: float = None):
        self.feature_name = feature_name
        self.feature_id = feature_id
        self.sign = sign
        self.thresholds = [-np.inf, threshold] if sign == Sign.seq else [threshold, np.inf]
        self.sample = sample
        self.value = value

    def __str__(self):
        if self.value is not None:
            return "The {} was {} for the {}, namely {}.".\
                format(self.feature_str(), self.signstr(), self.sampstr(), round(self.value, 1))
        else:
            return "The {} was {} for the {}.".\
                format(self.feature_str(), self.signstr(), self.sampstr())

    def aggregate(self, other: Decision) -> Decision:
        """
        Combine two decisions about the same feature if possible; use the most precise one.
        Assumes that it's about the same sample/foil and value.
        If combination is not possible, return None.
        """
        if other.feature_id != self.feature_id:
            return None

        new_sign = self.sign if other.sign == self.sign else Sign.between
        low = max(self.thresholds[0], other.thresholds[0])
        high = min(self.thresholds[1], other.thresholds[1])
        new_thresholds = [low, high]

        return Decision(self.feature_name, self.feature_id, new_sign, new_thresholds,
                        self.sample, self.value)

    def str_apply(self, value: float) -> str:
        """
        See whether the condition of this node holds for the value, and print the result.
        """
        passed = self.apply(value)
        if passed:
            return None
        else:
            return "The {} of the sample is not {}; it's {}.".\
                format(self.feature_str(), self.signstr(), value)

    def apply(self, value: float) -> bool:
        """
        Check whether the value passes the rule of this decision node.
        """
        return self.thresholds[0] < value <= self.thresholds[1]

    def signstr(self):
        if self.sign == Sign.seq:
            return "smaller or equal to {}".format(round(self.thresholds[1], 1))
        elif self.sign == Sign.between:
            return "between {} and {}".format(round(self.thresholds[0], 1),
                                              round(self.thresholds[1], 1))
        else:
            return "larger than {}".format(round(self.thresholds[0], 1))

    def sampstr(self):
        return "sample" if self.sample else "foil"

    def feature_str(self):
        return self.feature_name.replace("_", " ")

    def __eq__(self, other):
        """
        This is not a real equality, it checks for reverse rules.
        """
        if isinstance(other, Decision):
            if self.feature_name == other.feature_name:
                if self.thresholds == other.thresholds:
                    return True
                if self.sign != other.sign and (self.thresholds[0] == other.thresholds[1] or
                                                self.thresholds[1] == other.thresholds[0]):
                    return True
        return False
