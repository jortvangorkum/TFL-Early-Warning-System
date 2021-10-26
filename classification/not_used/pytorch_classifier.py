from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
import pandas
from pathlib import Path
from sklearn.model_selection._split import train_test_split
import numpy as np
from classification.network import Network


class NetworkClassifier(object):

    def __init__(self):
        self.get_data()

    def get_data(self):
        path = str(Path(__file__).parent.parent.joinpath("data", "data-normalization-output.csv"))
        df = pandas.read_csv(path)

        df.drop(df.columns[0], axis=1, inplace=True)
        avg_avg_score = df['average_score'].mean()
        df['average_score'] = df['average_score'].replace(np.nan, avg_avg_score)
        df.fillna(0, inplace=True)

        targets = df[['final_result__Distinction', 'final_result__Fail', 'final_result__Pass',
                      'final_result__Withdrawn']]
        df = df.drop(['final_result__Distinction', 'final_result__Fail', 'final_result__Pass',
                      'final_result__Withdrawn'], axis=1)

        x_train, x_test, y_train, y_test = train_test_split(df, targets, test_size=0.2,
                                                            random_state=32, shuffle=True,
                                                            stratify=targets)
        train_data = TensorDataset(Tensor(x_train.values), Tensor(y_train.values))
        self.train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

        test_data = TensorDataset(Tensor(x_test.values), Tensor(y_test.values))
        self.test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

        self.network = Network()

    def train(self, epochs: int):
        self.network.train(self.train_loader, epochs)

    def test(self):
        self.network.test(self.test_loader)
