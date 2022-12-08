import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input = nn.Linear(4, 128)
        self.encode1 = nn.Linear(128, 2048)
        self.encode2 = nn.Linear(2048, 512)
        self.decode1 = nn.Linear(512, 8)
        self.decode2 = nn.Linear(8, 1)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(.4)
        self.final = nn.Sigmoid()


    def forward(self, X, Y):

        out = self.input(X)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.encode1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.encode2(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.decode1(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.decode2(out)
        # out = self.activation(out)
        # out = self.dropout(out)

        out = self.final(out)

        return out


class LinearModelLoss(nn.Module):
    def __init__(self):
        super(LinearModelLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_orig, y_orig, y_pred):
        return self.mse_loss(y_pred, y_orig)
