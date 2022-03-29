import torch.nn as nn
import torch

from src.model.rnn import TimeLSTM
from src.model.attention import AttentionHawkes

class HYPHEN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        bs,
        attn_type="hawkes",
        do_hyp_lstm=False,
        learnable_curvature=False,
        init_curvature_val=0.0,
        n_class=2,
        time_param=True,
    ):
        if attn_type not in ["vanilla", "hawkes", "hyp_hawkes"]:
            raise ValueError(" Attn not of correct type")
        super().__init__()
        # self.hyp_lstm = TimeLSTMHyp(input_size, hidden_size)
        self.time_lstm = TimeLSTM(input_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, n_class)
        self.dropout = nn.Dropout(0.3)
        self.do_hyp_lstm = do_hyp_lstm
        self.time_param = time_param
        if learnable_curvature:
            # print("Init")
            self.c = torch.nn.Parameter(torch.tensor([init_curvature_val]).to("cuda"))
            # self.c =
        else:
            self.c = torch.FloatTensor([init_curvature_val]).to("cuda")
        self.attn_type = attn_type
        self.hidden_size = hidden_size
        # if attn_type == "hawkes":
        self.attention = AttentionHawkes(
            hidden_size, bs
        )  # Hawkes and temporal attn
        # elif attn_type == "hyp_hawkes":
        #     self.attention = HypHawkes(
        #         hidden_size, bs, c=self.c
        #     )  # Hawkes and temporal attn
        # else:
        #     self.attention = Attention(hidden_size)
        # self.cell_source = MobiusGRU(hidden_size, hidden_size, 1, k=self.c).to("cuda")
        self.normal_gru = nn.GRU(hidden_size, hidden_size, 1)

    def init_hidden(self, bs):
        h = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")
        c = (torch.zeros(bs, self.hidden_size, requires_grad=True)).to("cuda")

        return (h, c)

    def forward(self, inputs, timestamps, timestamps_inv):
        bs = inputs.shape[0]
        h_init, c_init = self.init_hidden(bs)
        if not self.time_param:
            shape = timestamps_inv.shape
            timestamps_inv = torch.ones(shape).to("cuda")
            timestamps = torch.zeros(shape).to("cuda")

        output, (_, _) = self.time_lstm(inputs, timestamps_inv)
        context, output = self.normal_gru(output.permute(1, 0, 2))

        output = output.permute(1, 0, 2)
        context = context.permute(1, 0, 2)

        output_fin, _ = self.attention(output, context, timestamps, self.c)

        output_fin = output_fin.permute(1, 0, 2)
        output_fin = output_fin.squeeze(0)

        output_fin = self.linear1(output_fin)
        output_fin = nn.relu(output_fin)
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