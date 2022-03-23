import torch

class MinMaxScaler():

    def __init__(self, x, device):
        x = torch.tensor(x)
        self.x_min = torch.min(x).to(device)
        self.x_max = torch.max(x).to(device)
        self.device = device

    def transform(self, x):
        return ((torch.tensor(x).to(self.device) - self.x_min)*100)/(self.x_max - self.x_min)

    def inverse_transform(self, out):
        return out*(self.x_max - self.x_min) + self.x_min
