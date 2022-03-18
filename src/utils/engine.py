import torch.nn as nn
import torch

class Engine:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for x, t, y in data_loader:
            self.optimizer.zero_grad()
            inputs = x.to(self.device)
            timestamps = t.to(self.device) 
            targets = y.to(self.device)

            outputs = self.model(inputs, timestamps).squeeze(1)
            loss = self.loss_fn(targets, outputs)

            # print(loss.shape)
            loss.backward()
            self.optimizer.step()

            final_loss += loss.item()

        return (final_loss/len(data_loader))

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        with torch.no_grad():
            for x, t, y in data_loader:
                inputs = x.to(self.device)
                timestamps = t.to(self.device)
                targets = y.to(self.device)

                outputs = self.model(inputs, timestamps).squeeze(1)
                loss = self.loss_fn(targets, outputs)

                final_loss += loss.item()

        return (final_loss/len(data_loader))