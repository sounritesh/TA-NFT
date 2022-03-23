import torch.nn as nn
import torch

from src.model.rnn import TimeLSTM

class TimeLSTM_MLP(nn.Module):
    def __init__(self, params):
        super(TimeLSTM_MLP, self).__init__()

        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.lstm_hidden_size = params['lstm_hidden_size']
        self.ntargets = params['ntargets']
        self.dropout = params['dropout']
     
        self.lstm = TimeLSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size)

        layers = [
            nn.Linear(self.lstm_hidden_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ntargets),
            # nn.Sigmoid()
        ]

        # for i in range(num_layers):
        #   layers.append(nn.Linear(hidden_size, hidden_size))
        #   layers.append(nn.Dropout(dropout))
        #   layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, timestamps):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out = self.lstm(x, timestamps)
        out = out[:,-1,:]
        out = self.mlp(out)
        return out


class LSTM_MLP(nn.Module):
    def __init__(self, params):
        super(LSTM_MLP, self).__init__()

        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.lstm_hidden_size = params['lstm_hidden_size']
        self.ntargets = params['ntargets']
        self.dropout = params['dropout']
        self.device = params['device']
     
        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.lstm_hidden_size, num_layers=1, batch_first=True)

        layers = [
            nn.Linear(self.lstm_hidden_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ntargets),
            # nn.Sigmoid()
        ]

        # for i in range(num_layers):
        #   layers.append(nn.Linear(hidden_size, hidden_size))
        #   layers.append(nn.Dropout(dropout))
        #   layers.append(nn.ReLU())

        self.mlp = nn.Sequential(*layers)

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.lstm_hidden_size, requires_grad=True)).to(self.device)
        c = (torch.zeros(bs, self.lstm_hidden_size, requires_grad=True)).to(self.device)

        return (h, c)

    def init_hidden_normal(self, bs):
      h = (torch.zeros(1, bs, self.lstm_hidden_size, requires_grad=True)).to(self.device)
      c = (torch.zeros(1, bs, self.lstm_hidden_size, requires_grad=True)).to(self.device)

      return (h, c)

    def forward(self, x):
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        bs, seq, embed = x.shape
        h0, c0 = self.init_hidden_normal(bs)

        out, _ = self.lstm(x, (h0, c0))
        out = out[:,-1,:]
        out = self.mlp(out)
        return out


class MLP(nn.Module):
    def __init__(self, params):
        super(MLP, self).__init__()

        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.dropout = params['dropout']
        self.ntargets = params['ntargets']

        layers = [
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.ntargets),
            # nn.Sigmoid()
        ]

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x[:, 0, :])