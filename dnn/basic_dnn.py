import torch.nn as nn


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.input = nn.Linear(4, 8)
        self.encode = nn.Linear(8, 16)
        self.decode = nn.Linear(16, 1)
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(.5)
        self.softmax = nn.Softmax()


    def forward(self, X, Y):

        out = self.input(X)
        out = self.activation(out)

        out = self.encode(out)
        out = self.activation(out)
        out = self.dropout(out)

        out = self.decode(out)
        #out = self.activation(out)

        return out


class LinearModelLoss(nn.Module):
    def __init__(self):
        super(LinearModelLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, x_orig, y_orig, y_pred):
        return self.mse_loss(y_pred, y_orig)
