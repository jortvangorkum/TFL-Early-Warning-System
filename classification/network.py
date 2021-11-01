from torch import nn, max
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.build_network()

    def build_network(self):
        self.network = nn.Sequential(
            nn.Linear(53, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Softmax(dim=1)
            )

        self.optimizer = optim.SGD(self.parameters(), lr=0.0001, momentum=0.9)
        # self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, x):
        output = self.network(x)
        return output

    def train(self, dataloader: DataLoader, epochs: int):
        self.criterion = nn.BCEWithLogitsLoss()

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(dataloader):
                inputs, labels = data

                self.optimizer.zero_grad()

                outputs = self(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                clip_grad_norm_(self.parameters(), 1)
                self.optimizer.step()

                running_loss += loss.item()
                if i % 500 == 499:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 500))
                    running_loss = 0.0

        print("done training")

    def test(self, dataloader: DataLoader):
        correct = 0
        total = 0
        for i, data in enumerate(dataloader):
            inputs, labels = data
            outputs = self(inputs)
            _, predicted = max(outputs, 1)
            total += labels.size(0)
            correct += sum(labels[i][predicted[i]] != 0 for i, _ in enumerate(labels))

        print("Accuracy on test: ", correct / total)
