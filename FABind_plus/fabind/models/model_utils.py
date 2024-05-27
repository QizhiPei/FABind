# Attention Module Adapted From https://github.com/aqlaboratory/openfold/blob/main/openfold/model/primitives.py
import math
from typing import List, Tuple, Optional

import torch
from torch.nn import Linear, LayerNorm
from torch.nn.functional import softmax
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, out_channels)
        
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        if self.args.use_ln_mlp:
            z = self.layernorm(z)
        
        if self.args.dropout > 0:
            z = self.linear2(self.dropout(self.linear1(z).relu()))
        else:
            z = self.linear2(self.linear1(z).relu())
        return z

class MLPwithLastAct(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout1 = nn.Dropout(args.dropout)
            self.dropout2 = nn.Dropout(args.dropout)
        self.linear1 = Linear(embedding_channels, int(n*embedding_channels))
        self.linear2 = Linear(int(n*embedding_channels), out_channels)
        
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        if self.args.use_ln_mlp:
            z = self.layernorm(z)
            
        if self.args.dropout > 0:
            z = self.dropout2(self.linear2(self.dropout1(self.linear1(z).relu())).relu())
        else:
            z = self.linear2(self.linear1(z).relu()).relu()
        return z

class MLPwoBias(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.dropout > 0:
            self.dropout = nn.Dropout(args.dropout)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, out_channels, bias=False)
        
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        if self.args.use_ln_mlp:
            z = self.layernorm(z)
        if self.args.dropout > 0:
            z = self.linear2(self.dropout(self.linear1(z).relu()))
        else:
            z = self.linear2(self.linear1(z).relu())
        return z


class MLP4Confidence(nn.Module):
    def __init__(self, args, embedding_channels=256, out_channels=256, n=4):
        super().__init__()
        self.args = args
        if self.args.confidence_use_ln_mlp:
            self.layernorm = nn.LayerNorm(embedding_channels)
        if args.confidence_dropout > 0:
            self.dropout = nn.Dropout(args.confidence_dropout)
        self.linear1 = Linear(embedding_channels, n*embedding_channels)
        self.linear2 = Linear(n*embedding_channels, out_channels)
        
    def forward(self, z):
        # z of shape b, i, j, embedding_channels, where i is protein dim, j is compound dim.
        if self.args.confidence_use_ln_mlp:
            z = self.layernorm(z)
        
        if self.args.confidence_dropout > 0:
            z = self.linear2(self.dropout(self.linear1(z).relu()))
        else:
            z = self.linear2(self.linear1(z).relu())
        return z


def permute_final_dims(tensor: torch.Tensor, inds: List[int]):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds]).contiguous()

def flatten_final_dims(t: torch.Tensor, no_dims: int):
    return t.reshape(t.shape[:-no_dims] + (-1,))


def _attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, biases: List[torch.Tensor], distance=None, dis_pair_type=None, mha_permu=False) -> torch.Tensor:
    # [*, H, Q, C_hidden]
    query = permute_final_dims(query, (1, 0, 2))
    # [*, H, C_hidden, K]
    key = permute_final_dims(key, (1, 2, 0))
    # [*, H, V, C_hidden]
    value = permute_final_dims(value, (1, 0, 2))
    # [*, H, Q, K]
    a = torch.matmul(query, key) # [B, ]
    for b in biases:
        a = a + b
    if dis_pair_type == 'add':
        # mask_value = 0.0
        # epsilon = 1e-5
        # safe_inverse_distance = torch.where(distance == 0.0, torch.full_like(distance, mask_value), 1.0 / (distance + epsilon))
        safe_inverse_distance = distance
        if mha_permu:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 0, 1))
        else:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 1, 0))
        a = a + safe_inverse_distance
    a = softmax(a, -1)
    if dis_pair_type == 'mul':
        # mask_value = 0.0
        # epsilon = 1e-5
        # safe_inverse_distance = torch.where(distance == 0.0, torch.full_like(distance, mask_value), 1.0 / (distance + epsilon))
        safe_inverse_distance = distance
        if mha_permu:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 0, 1))
        else:
            safe_inverse_distance = permute_final_dims(safe_inverse_distance, (2, 1, 0))
        a = a * safe_inverse_distance
    # [*, H, Q, C_hidden]
    a = torch.matmul(a, value)
    # [*, Q, H, C_hidden]
    a = a.transpose(-2, -3)

    return a


class Attention(nn.Module):
    """
    Standard multi-head attention using AlphaFold's default layer
    initialization. Allows multiple bias vectors.
    """

    def __init__(
            self,
            args,
            c_q: int,
            c_k: int,
            c_v: int,
            c_hidden: int,
            no_heads: int,
            gating: bool = True,
            mha_permu=False,
    ):
        """
        Args:
            c_q:
                Input dimension of query data
            c_k:
                Input dimension of key data
            c_v:
                Input dimension of value data
            c_hidden:
                Per-head hidden dimension
            no_heads:
                Number of attention heads
            gating:
                Whether the output should be gated using query data
        """
        super(Attention, self).__init__()

        self.args = args
        self.mha_permu = mha_permu
        
        self.c_q = c_q
        self.c_k = c_k
        self.c_v = c_v
        self.c_hidden = c_hidden
        self.no_heads = no_heads
        self.gating = gating

        # DISCREPANCY: c_hidden is not the per-head channel dimension, as
        # stated in the supplement, but the overall channel dimension.

        self.linear_q = Linear(self.c_q, self.c_hidden * self.no_heads, bias=False)
        self.linear_k = Linear(self.c_k, self.c_hidden * self.no_heads, bias=False)
        self.linear_v = Linear(self.c_v, self.c_hidden * self.no_heads, bias=False)
        self.linear_o = Linear(self.c_hidden * self.no_heads, self.c_q)

        self.linear_g = None
        if self.gating:
            self.linear_g = Linear(
                self.c_q, self.c_hidden * self.no_heads
            )

        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self,
                  q_x: torch.Tensor,
                  kv_x: torch.Tensor
                  ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # [*, Q/K/V, H * C_hidden]
        q = self.linear_q(q_x)
        k = self.linear_k(kv_x)
        v = self.linear_v(kv_x)

        # [*, Q/K, H, C_hidden]
        q = q.view(q.shape[:-1] + (self.no_heads, -1))
        k = k.view(k.shape[:-1] + (self.no_heads, -1))
        v = v.view(v.shape[:-1] + (self.no_heads, -1))

        q /= math.sqrt(self.c_hidden)
 
        return q, k, v

    def _wrap_up(self,
                 o: torch.Tensor,
                 q_x: torch.Tensor
                 ) -> torch.Tensor:
        if (self.linear_g is not None):
            g = self.sigmoid(self.linear_g(q_x))

            # [*, Q, H, C_hidden]
            g = g.view(g.shape[:-1] + (self.no_heads, -1))
            o = o * g

        # [*, Q, H * C_hidden]
        o = flatten_final_dims(o, 2)

        # [*, Q, C_q]
        o = self.linear_o(o)

        return o

    def forward(
            self,
            q_x: torch.Tensor,
            kv_x: torch.Tensor,
            biases: Optional[List[torch.Tensor]] = None,
            distance=None,
    ) -> torch.Tensor:
        """
        Args:
            q_x:
                [*, Q, C_q] query data
            kv_x:
                [*, K, C_k] key data
            biases:
                List of biases that broadcast to [*, H, Q, K]
        Returns
            [*, Q, C_q] attention update
        """
        if biases is None:
            biases = []

        q, k, v = self._prep_qkv(q_x, kv_x)
        o = _attention(q, k, v, biases, distance=distance, dis_pair_type=self.args.rel_dis_pair_bias, mha_permu=self.mha_permu)
        o = self._wrap_up(o, q_x)

        return o


class InteractionModule(torch.nn.Module):
    # TODO: test opm False and True
    def __init__(self, node_hidden_dim, pair_hidden_dim, hidden_dim, opm=False, rm_layernorm=False):
        super(InteractionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim
        self.opm = opm

        self.rm_layernorm = rm_layernorm
        if not rm_layernorm:
            self.layer_norm_p = nn.LayerNorm(node_hidden_dim)
            self.layer_norm_c = nn.LayerNorm(node_hidden_dim)

        if self.opm:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim ** 2, pair_hidden_dim)
        else:
            self.linear_p = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_c = nn.Linear(node_hidden_dim, hidden_dim)
            self.linear_out = nn.Linear(hidden_dim, pair_hidden_dim)

    def forward(self, p_embed, c_embed,
                p_mask=None, c_mask=None):
        # mask
        if p_mask is None:
            p_mask = p_embed.new_ones(p_embed.shape[:-1], dtype=torch.bool)
        if c_mask is None:
            c_mask = c_embed.new_ones(c_embed.shape[:-1], dtype=torch.bool)
        inter_mask = torch.einsum("...i,...j->...ij", p_mask, c_mask)  # (Np, Nc)

        if not self.rm_layernorm:
            p_embed = self.layer_norm_p(p_embed)  # (Np, C_node)
            c_embed = self.layer_norm_c(c_embed)  # (Nc, C_node)
        if self.opm:
            p_embed = self.linear_p(p_embed)  # (Np, C_hidden)
            c_embed = self.linear_c(c_embed)  # (Nc, C_hidden)
            inter_embed = torch.einsum("...bc,...de->...bdce", p_embed, c_embed)
            inter_embed = torch.flatten(inter_embed, -2) # vecterize last two dim
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        else:
            p_embed = self.linear_p(p_embed)  # (Np, C_hidden)
            c_embed = self.linear_c(c_embed)  # (Nc, C_hidden)
            inter_embed = torch.einsum("...ik,...jk->...ijk", p_embed, c_embed)
            inter_embed = self.linear_out(inter_embed) * inter_mask.unsqueeze(-1)
        return inter_embed, inter_mask



class GaussianSmearing(torch.nn.Module):
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist[..., None] - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


@torch.jit.script
def gaussian(x, mean, std):
    pi = 3.14159
    a = (2*pi) ** 0.5
    return torch.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)
class GaussianLayer(nn.Module):
    def __init__(self, K=128):
        super().__init__()
        self.K = K
        self.means = nn.Embedding(1, K)
        self.stds = nn.Embedding(1, K)
        nn.init.uniform_(self.means.weight, 0, 3)
        nn.init.uniform_(self.stds.weight, 0, 3)

    def forward(self, x):
        x = x.expand(-1, -1, -1, self.K)
        mean = self.means.weight.float().view(-1)
        std = self.stds.weight.float().view(-1).abs() + 1e-2
        return gaussian(x.float(), mean, std).type_as(self.means.weight)