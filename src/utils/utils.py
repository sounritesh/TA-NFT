import torch

class MinMaxScaler():

    def __init__(self, x, device):
        x = torch.tensor(x)
        self.x_min = torch.min(x).to(device)
        self.x_max = torch.max(x).to(device)

    def transform(self, x):
        return (x - self.x_min)/(self.x_max - self.x_min)

    def inverse_transform(self, out):
        return torch.tensor(out)*(self.x_max - self.x_min) + self.x_min
