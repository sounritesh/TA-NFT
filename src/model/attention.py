import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as f

class AttentionHawkes(torch.nn.Module):
    def __init__(self, dimensions, bs, attention_type="general"):
        super(AttentionHawkes, self).__init__()
        print(attention_type)
        if attention_type not in ["dot", "general"]:
            raise ValueError("Invalid attention type selected.")

        self.attention_type = attention_type
        if self.attention_type == "general":
            self.linear_in = torch.nn.Linear(dimensions, dimensions, bias=False)

        self.linear_out = torch.nn.Linear(dimensions * 2, dimensions, bias=False)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.tanh = torch.nn.Tanh()
        #self.ae = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))
        #self.ab = torch.nn.Parameter(torch.FloatTensor(bs, 1, 1))
        self.ae = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.ab = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)


    def forward(self, query, context, delta_t, c=1.0):
        batch_size, output_len, dimensions = query.size()
        #print(dimensions)
        query_len = context.size(1)
        if self.attention_type == "general":
            query = query.reshape(batch_size * output_len, dimensions)
            query = self.linear_in(query)
            query = query.reshape(batch_size, output_len, dimensions)

        attention_scores = torch.bmm(query, context.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size * output_len, query_len)
        attention_weights = self.softmax(attention_scores)
        attention_weights = attention_weights.view(batch_size, output_len, query_len)

        if output_len==1:
            mix = attention_weights * (context.permute(0, 2, 1))
        else:
            mix = torch.matmul(context.permute(0, 2, 1), attention_weights.permute(0, 2, 1))
            #print("Mix:", mix.shape)
        #print("Ab: ", torch.sum(self.ab))        
        bt = torch.exp(-1 * self.ab * delta_t.reshape(batch_size, query_len,  1).permute(0, 2, 1))
        #print(torch.sum(bt))
        term_2 = nn.ReLU()(self.ae * mix * bt)
        if output_len==1:
            mix = torch.sum(term_2 + mix, -1).unsqueeze(1)
        else:
            mix = torch.cumsum(term_2 + mix, -1).transpose(1,2)
        #print("Mix:", mix.shape)
        #print("Query:", query.shape)
        combined = torch.cat((mix, query), dim=2)
        combined = combined.view(batch_size * output_len, 2 * dimensions)
        output = self.linear_out(combined).view(batch_size, output_len, dimensions)
        output = self.tanh(output)
        return output, attention_weights


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention_Vanilla(nn.Module):
    def __init__(self, num_heads, dimensions, bs, attention_type='general'):
        super().__init__()
        dim_in, dim_q, dim_k = dimensions, dimensions, dimensions
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, delta_t: Tensor) -> Tensor:
        value = query
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )