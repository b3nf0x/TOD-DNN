import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input = nn.Linear(4, 20)
        self.encode = nn.Linear(20, 256)
        self.decode = nn.Linear(256, 1)
        self.activation = nn.LeakyReLU()


    def forward(self, X, Y):
        out = self.input(X)
        out = self.activation(out)
        out = self.encode(out)
        out = self.decode(out)
        out = self.activation(out)
        return out


class LinearModelLoss(nn.Module):
    def __init__(self):
        super(LinearModelLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_orig, y_orig, y_pred):
        return self.mse_loss(y_pred, y_orig)
