from pytorch_classifier import NetworkClassifier
import torch
from pathlib import Path
from tf_network_classifier import TFNetwork
from naive_bayes_classifier import NaiveBayes


def run_pytorch():
    """
    Run the pytorch model. Not working well atm
    """
    network = NetworkClassifier()
    network.train(50)
    data_folder = str(Path(__file__).parent.parent.joinpath("data", "network.pt"))
    torch.save(network.network.state_dict(), data_folder)
    # network.network.load_state_dict(torch.load(data_folder))

    network.test()


def run_tensorflow():
    """
    Run the tensorflow model. Doing kind of alright
    """
    tf_network = TFNetwork()
    tf_network.initialize()


def run_naive_bayes():
    """
    Run the naive bayes classifier. Haven't tried different flavours yet so performance can
    probably be better than this.
    """
    nbc = NaiveBayes()
    nbc.run_model()


if __name__ == '__main__':
    """
    Run one of the classifiers.
    """
#     run_pytorch()
#     run_tensorflow()
    run_naive_bayes()
