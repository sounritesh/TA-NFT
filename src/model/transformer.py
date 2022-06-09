import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as f
from src.model.attention import AttentionHawkes
from src.utils.config import DEVICE, UTC

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dimensions, bs, attention_type="general"):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHawkes(dimensions, bs, attention_type) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dimensions, dimensions)

    def forward(self, query, context, delta_t, c=1.0) -> Tensor:
        return self.linear(
            torch.cat([h(query, context, delta_t, c)[0] for h in self.heads], dim=-1)
        )


def position_encoding(seq_len: int, dim_model: int) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=DEVICE).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=DEVICE).reshape(1, 1, -1)
    #phase = pos / (1e4 ** (dim // dim_model))
    phase = pos / (1e4 ** torch.div(dim, dim_model, rounding_mode='floor'))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))

def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.ReLU(),
        nn.Linear(dim_feedforward, dim_input),
    )

class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        bs,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        attention_type = "general"
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, bs, attention_type),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src, timestamps, timestamps_inv, reach_weights) -> Tensor:
        #print("Timestamps inv: ", timestamps_inv.shape)
        #context = src * timestamps_inv.unsqueeze(dim=-1)
        #print("Src:", torch.sum(src))
        #print("RW:", torch.sum(reach_weights))
        #context = src * reach_weights.unsqueeze(dim=-1)
        context = src * timestamps_inv.unsqueeze(dim=-1)
        src = self.attention(src, context, timestamps)
        return self.feed_forward(src)