import torch.nn as nn
import torch

class Engine:
    def __init__(self, model, optimizer, device, model_type):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCEWithLogitsLoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()
            inputs = data['encs'].to(self.device)
            timestamps = data['ts_w'].to(self.device) 
            targets = data['price'].to(self.device)
            
            print(inputs.size(), timestamps.size(), targets.size())

            if self.model_type == 'tlstm':
                outputs = self.model(inputs, timestamps).squeeze(1)
            else:
                outputs = self.model(inputs).squeeze(1)
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
            for data in data_loader:
                inputs = data['encs'].to(self.device)
                timestamps = data['ts_w'].to(self.device) 
                targets = data['price'].to(self.device)

                if self.model_type == 'tlstm':
                    outputs = self.model(inputs, timestamps).squeeze(1)
                else:
                    outputs = self.model(inputs).squeeze(1)
                loss = self.loss_fn(targets, outputs)

                final_loss += loss.item()

        return (final_loss/len(data_loader))