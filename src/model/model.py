from pytest import param
import torch.nn as nn
import torch
from torch import Tensor
from src.model.rnn import TimeLSTM, RTimeLSTM
from src.model.attention import AttentionHawkes
from src.model.transformer import TransformerEncoderLayer, position_encoding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        params,
        bs,
        num_layers: int = 2,
        num_heads: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn_type = "hawkes"
        self.hidden_size = params["hidden_size"]
        self.bs = bs
        self.linear1 = nn.Linear(params['input_size'], self.hidden_size)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(self.bs, self.hidden_size, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src, timestamps, timestamps_inv, reach_weights) -> Tensor:
        src = self.linear1(src)
        seq_len, dimension = src.size(1), src.size(2)
        src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src, timestamps, timestamps_inv, reach_weights)

        #print("SRC: ", src.shape)
        #print("Reach Weights: ", reach_weights.shape)
        src = torch.sum(src * timestamps_inv.unsqueeze(dim=-1), 1, keepdim=True)
        return src

class TLSTM_Hawkes(nn.Module):
    def __init__(
        self,
        params,
        bs
    ):
        super().__init__()
        # self.hyp_lstm = TimeLSTMHyp(input_size, hidden_size)
        self.time_lstm = TimeLSTM(params["input_size"], params["hidden_size"])
        self.linear1 = nn.Linear(params["hidden_size"], params["hidden_size"])
        self.linear2 = nn.Linear(params["hidden_size"], params["ntargets"])
        self.dropout = nn.Dropout(params["dropout"])
        self.attn_type = "hawkes"
        self.hidden_size = params["hidden_size"]
        self.bs = bs
        self.attention = AttentionHawkes(
            self.hidden_size, self.bs
        )  # Hawkes and temporal attn

        self.normal_gru = nn.GRU(self.hidden_size, self.hidden_size, 1)

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")
        c = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")

        return (h, c)

    def forward(self, inputs, timestamps, timestamps_inv):
        bs = inputs.shape[0]
        h_init, c_init = self.init_hidden(bs)

        output = self.time_lstm(inputs, timestamps_inv)
        context, output = self.normal_gru(output.permute(1, 0, 2))

        output = output.permute(1, 0, 2)
        context = context.permute(1, 0, 2)

        output_fin, attention_weights = self.attention(output, context, timestamps)

        output_fin = output_fin.permute(1, 0, 2)
        output_fin = output_fin.squeeze(0)

        output_fin = self.linear1(output_fin)
        output_fin = nn.ReLU()(output_fin)
        output_fin = self.dropout(output_fin)
        output_fin = self.linear2(output_fin)
        #print("output fin:", output_fin.shape)
        return output_fin

class RTLSTM_Hawkes(nn.Module):
    def __init__(
        self,
        params,
        bs
    ):
        super().__init__()
        # self.hyp_lstm = TimeLSTMHyp(input_size, hidden_size)
        self.time_lstm = RTimeLSTM(params["input_size"], params["hidden_size"])
        self.linear1 = nn.Linear(params["hidden_size"], params["hidden_size"])
        self.linear2 = nn.Linear(params["hidden_size"], params["ntargets"])
        self.dropout = nn.Dropout(params["dropout"])
        self.attn_type = "hawkes"
        self.hidden_size = params["hidden_size"]
        self.bs = bs
        self.attention = AttentionHawkes(
            self.hidden_size, self.bs
        )  # Hawkes and temporal attn

        self.normal_gru = nn.GRU(self.hidden_size, self.hidden_size, 1)

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")
        c = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")

        return (h, c)

    def forward(self, inputs, timestamps, timestamps_inv, reach_weights):
        bs = inputs.shape[0]
        h_init, c_init = self.init_hidden(bs)

        output = self.time_lstm(inputs, timestamps_inv, reach_weights)
        context, output = self.normal_gru(output.permute(1, 0, 2))

        output = output.permute(1, 0, 2)
        context = context.permute(1, 0, 2)

        output_fin, attention_weights = self.attention(output, context, timestamps)

        output_fin = output_fin.permute(1, 0, 2)
        output_fin = output_fin.squeeze(0)

        output_fin = self.linear1(output_fin)
        output_fin = nn.ReLU()(output_fin)
        output_fin = self.dropout(output_fin)
        output_fin = self.linear2(output_fin)
        return output_fin

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

        if params['classification']:
            layers.append(nn.Sigmoid())

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

        if params['classification']:
            layers.append(nn.Sigmoid())

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
        
        if params['classification']:
            layers.append(nn.Sigmoid())
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):

        return self.mlp(x[:, 0, :])