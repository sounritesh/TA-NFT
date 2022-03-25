import torch.nn as nn
import torch
from tqdm import tqdm

class Engine:
    def __init__(self, model, optimizer, device, model_type, scaler=None, classification=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model_type = model_type
        self.scaler =  scaler
        self.classification = classification

    @staticmethod
    def loss_fn(targets, outputs):
        return nn.BCELoss()(outputs, targets)
        # return nn.MSELoss()(outputs, targets)

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
                outputs = self.model(inputs, timestamps)
            else:
                outputs = self.model(inputs)

            # outputs = self.scaler.inverse_transform(outputs)
            print(outputs.shape)
            if self.classification:
                loss = self.loss_fn(targets, outputs.squeeze(1))
            else:
                loss = self.mse_loss(targets, outputs.squeeze(1))

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
                targets = data['price'].to(self.device)

                if self.model_type == 'tlstm':
                    outputs = self.model(inputs, timestamps)
                else:
                    outputs = self.model(inputs)

                # outputs = self.scaler.inverse_transform(outputs)
                if self.classification:
                    loss = self.loss_fn(targets, outputs.squeeze(1))
                else:
                    loss = self.mse_loss(targets, outputs.squeeze(1))

                final_loss += loss.item()

        return (final_loss/len(data_loader)),