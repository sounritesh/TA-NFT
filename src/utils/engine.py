import torch.nn as nn
import torch
from tqdm import tqdm

class Engine:
    def __init__(self, model, optimizer, device, model_type, scaler):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.scaler =  scaler

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCELoss()(outputs, targets)

    @staticmethod
    def mse_loss(targets, outputs):
        return nn.MSELoss()(outputs, targets)

    def train(self, data_loader):
        self.model.train()
        final_loss = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            self.optimizer.zero_grad()
            inputs = data['encs'].to(self.device)
            timestamps = data['ts_w'].to(self.device) 
            targets = data['price'].to(self.device)
            
            # print(inputs.size(), timestamps.size(), targets.size())

            if self.model_type == 'tlstm':
                outputs = self.model(inputs, timestamps).squeeze(1)
            else:
                outputs = self.model(inputs).squeeze(1)

            # outputs = self.scaler.inverse_transform(outputs)
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
            for data in tqdm(data_loader, total=len(data_loader)):
                inputs = data['encs'].to(self.device)
                timestamps = data['ts_w'].to(self.device) 
                targets = data['price_og'].to(self.device)

                if self.model_type == 'tlstm':
                    outputs = self.model(inputs, timestamps).squeeze(1)
                else:
                    outputs = self.model(inputs).squeeze(1)

                outputs = self.scaler.inverse_transform(outputs)
                loss = self.mse_loss(targets, outputs)

                final_loss += loss.item()

        return (final_loss/len(data_loader))