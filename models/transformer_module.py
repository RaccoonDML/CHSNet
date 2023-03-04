import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
import copy


class GlobalMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim  # 512
        self.num_heads = num_heads  # 2
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads  # 256
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.in_proj_k = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.in_proj_v = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_q.weight)
        nn.init.xavier_uniform_(self.in_proj_k.weight)
        nn.init.xavier_uniform_(self.in_proj_v.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_q.bias is not None:
            nn.init.constant_(self.in_proj_q.bias, 0.)
            nn.init.constant_(self.in_proj_k.bias, 0.)
            nn.init.constant_(self.in_proj_v.bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, shape, value):
        # [hw b c] [hw b c] [hw b c]
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        q = self.in_proj_q(query)  # [hw b c] * [c c]-> [hw b c]
        k = self.in_proj_k(key)
        v = self.in_proj_v(value)
        q = q * self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)  # [hw b c] -> [hw b*4 c//4] -> [b*4 hw c//4]
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))  # [4*b hw c//4] * [4*b c//4 hw] -> [4b hw hw]
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)  # [4b hw softmax(hw)] * [4b hw c//4] -> [4b hw c//4]
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)  # [hw 4b c//4] -> [hw b c]
        attn = self.out_proj(attn)
        return attn


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = GlobalMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, shape, pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)  # src: hw b c
        src2 = self.self_attn(q, k, shape, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Transformer(nn.Module):
    def __init__(self, layers=4, dim=512, norm=None):
        super().__init__()
        d_model = dim
        nhead = 2
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"
        normalize_before = False

        self.layers = nn.ModuleList([copy.deepcopy(TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before))
                                     for i in range(layers)])
        self.norm = norm

    def forward(self, src, shape, pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(output, shape, pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

