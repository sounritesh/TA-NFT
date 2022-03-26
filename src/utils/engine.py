import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np

from src.utils.metric import classification_report

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
        final_prec = 0
        final_recall = 0
        final_fscore = 0
        final_mcc = 0
        final_acc = 0
        for data in tqdm(data_loader, total=len(data_loader)):
            self.optimizer.zero_grad()
            inputs = data['encs'].to(self.device)
            timestamps = data['ts_w'].to(self.device) 
            if self.classification:
                targets = data['target'].to(self.device)
            else:
                targets = data['price'].to(self.device)
            
            # print(inputs.size(), timestamps.size(), targets.size())

            if self.model_type == 'tlstm':
                outputs = self.model(inputs, timestamps).squeeze(1)
            else:
                outputs = self.model(inputs).squeeze(1)

            # outputs = self.scaler.inverse_transform(outputs)
            if self.classification:
                loss = self.loss_fn(targets, outputs)
            else:
                loss = self.mse_loss(targets, outputs)

            # print(loss.shape)
            loss.backward()
            self.optimizer.step()

            final_loss += loss.item()

            if self.classification:
                prec, recall, fscore, mcc, acc = classification_report(targets, outputs)
                final_prec += prec
                final_recall += recall
                final_fscore += fscore
                final_mcc += mcc
                final_acc += acc

                # print(f"Precision: {prec}; Recall: {recall}; F1-score: {fscore}; MCC: {mcc}; Accuracy: {acc};")

        return final_loss/len(data_loader), np.array([final_prec, final_recall, final_fscore, final_mcc, final_acc])/len(data_loader)

    def evaluate(self, data_loader):
        self.model.eval()
        final_loss = 0
        final_loss = 0
        final_prec = 0
        final_recall = 0
        final_fscore = 0
        final_mcc = 0
        final_acc = 0
        with torch.no_grad():
            for data in tqdm(data_loader, total=len(data_loader)):
                inputs = data['encs'].to(self.device)
                timestamps = data['ts_w'].to(self.device) 
                if self.classification:
                    targets = data['target'].to(self.device)
                else:
                    targets = data['price'].to(self.device)

                if self.model_type == 'tlstm':
                    outputs = self.model(inputs, timestamps).squeeze(1)
                else:
                    outputs = self.model(inputs).squeeze(1)

                # outputs = self.scaler.inverse_transform(outputs)
                if self.classification:
                    loss = self.loss_fn(targets, outputs)
                else:
                    loss = self.mse_loss(targets, outputs)

                final_loss += loss.item()

                if self.classification:
                    prec, recall, fscore, mcc, acc = classification_report(targets, outputs)
                    final_prec += prec
                    final_recall += recall
                    final_fscore += fscore
                    final_mcc += mcc
                    final_acc += acc

                    # print(f"Precision: {prec}; Recall: {recall}; F1-score: {fscore}; MCC: {mcc}; Accuracy: {acc};")

        return final_loss/len(data_loader), np.array([final_prec, final_recall, final_fscore, final_mcc, final_acc])/len(data_loader)