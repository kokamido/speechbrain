from torch import nn


class TestClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(
            nn.Conv1d(in_channels=1, out_channels=2, kernel_size=(2,), stride=(2,))
        )
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv1d(2, 4, (2,), (2,)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv1d(4, 8, (2,), (2,)))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Conv1d(8, 1, (3, )))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
