# import torch
# from pathlib import Path
# from classification.not_used.pytorch_classifier import NetworkClassifier
# from classification.not_used.tf_network_classifier import TFNetwork
# from classification.not_used.naive_bayes_classifier import NaiveBayes
from classification.tree_manager import TreeManager

# 
# def run_pytorch():
#     """
#     Run the pytorch model. Not working well atm
#     """
#     network = NetworkClassifier()
#     network.train(50)
#     data_folder = str(Path(__file__).parent.parent.joinpath("data", "network.pt"))
#     torch.save(network.network.state_dict(), data_folder)
#     # network.network.load_state_dict(torch.load(data_folder))
# 
#     network.test()


# def run_tensorflow():
#     """
#     Run the tensorflow model. Doing kind of alright
#     """
#     tf_network = TFNetwork()
#     tf_network.initialize()


# def run_naive_bayes():
#     """
#     Run the naive bayes classifier. Haven't tried different flavours yet so performance can
#     probably be better than this.
#     """
#     nbc = NaiveBayes()
#     nbc.run_model()


def run_decision_tree():
    """
    Run the decision tree classifier, explain a sample and plot the scores of the different trees.
    """
    tree_manager = TreeManager()
    tree_manager.create_trees([1, 2, 3])
    tree_manager.plot_scores()
    tree_manager.explain_sample(importance=1, months=6, index=5)


if __name__ == '__main__':
    """
    Run one of the classifiers.
    """
#     run_pytorch()
#     run_tensorflow()
#    run_naive_bayes()
    run_decision_tree()
