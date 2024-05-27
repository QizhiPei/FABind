import torch
from torch import nn
from torch.nn import LayerNorm, Linear

from models.model_utils import permute_final_dims, Attention, InteractionModule, MLP, MLPwithLastAct, MLPwoBias

class CrossAttentionModule(nn.Module):
    def __init__(self, args, node_hidden_dim, pair_hidden_dim, rm_layernorm=False, keep_trig_attn=False, dist_hidden_dim=32, normalize_coord=None):
        super().__init__()
        self.pair_hidden_dim = pair_hidden_dim
        self.keep_trig_attn = keep_trig_attn

        self.p_attention_block = RowAttentionBlock(args, node_hidden_dim, pair_hidden_dim, no_heads=args.mha_heads, rm_layernorm=rm_layernorm, mha_permu=True)
        self.c_attention_block = RowAttentionBlock(args, node_hidden_dim, pair_hidden_dim, no_heads=args.mha_heads, rm_layernorm=rm_layernorm, mha_permu=False)
        self.p_transition = MLPwithLastAct(args, embedding_channels=node_hidden_dim, n=args.mlp_hidden_scale, out_channels=node_hidden_dim)
        self.c_transition = MLPwithLastAct(args, embedding_channels=node_hidden_dim, n=args.mlp_hidden_scale, out_channels=node_hidden_dim)
        self.pair_transition = MLPwithLastAct(args, embedding_channels=pair_hidden_dim, n=args.mlp_hidden_scale, out_channels=pair_hidden_dim)
        self.inter_layer = InteractionModule(node_hidden_dim, pair_hidden_dim, 32, opm=False, rm_layernorm=rm_layernorm)

    def forward(self, 
                p_embed_batched, p_mask,
                c_embed_batched, c_mask,
                pair_embed, pair_mask,
                c_c_dist_embed=None, p_p_dist_embed=None,
                distance=None):
        
        p_embed_batched = self.p_attention_block(node_embed_i=p_embed_batched,
                                                 node_embed_j=c_embed_batched,
                                                 pair_embed=pair_embed,
                                                 pair_mask=pair_mask,
                                                 node_mask_i=p_mask,
                                                 distance=distance,)
        c_embed_batched = self.c_attention_block(node_embed_i=c_embed_batched,
                                                 node_embed_j=p_embed_batched,
                                                 pair_embed=pair_embed.transpose(-2, -3),
                                                 pair_mask=pair_mask.transpose(-1, -2),
                                                 node_mask_i=c_mask,
                                                 distance=distance,)
        p_embed_batched = p_embed_batched + self.p_transition(p_embed_batched) 
        c_embed_batched = c_embed_batched + self.c_transition(c_embed_batched)

        pair_embed = pair_embed + self.inter_layer(p_embed_batched, c_embed_batched, p_mask, c_mask)[0]

        pair_embed = self.pair_transition(pair_embed) * pair_mask.to(torch.float).unsqueeze(-1)
        return p_embed_batched, c_embed_batched, pair_embed



class RowAttentionBlock(nn.Module):
    inf = 1e9

    def __init__(self, args, node_hidden_dim, pair_hidden_dim, attention_hidden_dim=32, no_heads=4, dropout=0.1, rm_layernorm=False, mha_permu=False):
        super(RowAttentionBlock, self).__init__()
        self.no_heads = no_heads
        self.attention_hidden_dim = attention_hidden_dim
        self.pair_hidden_dim = pair_hidden_dim
        self.node_hidden_dim = node_hidden_dim

        self.rm_layernorm = rm_layernorm
        if not self.rm_layernorm:
            self.layernorm_node_i = LayerNorm(node_hidden_dim)
            self.layernorm_node_j = LayerNorm(node_hidden_dim)
            self.layernorm_pair = LayerNorm(pair_hidden_dim)

        self.linear = Linear(pair_hidden_dim, self.no_heads)
        self.linear_g = Linear(pair_hidden_dim, self.no_heads)

        self.dropout = nn.Dropout(args.dropout)

        self.mha = Attention(args, node_hidden_dim, node_hidden_dim, node_hidden_dim, attention_hidden_dim, no_heads, mha_permu=mha_permu)
    
    def forward(self, node_embed_i, node_embed_j, pair_embed, pair_mask, node_mask_i, distance=None):
        if not self.rm_layernorm:
            node_embed_i = self.layernorm_node_i(node_embed_i)  # (*, I, C_node)
            node_embed_j = self.layernorm_node_j(node_embed_j)  # (*, J, C_node)
            pair_embed = self.layernorm_pair(pair_embed)  # (*, I, J, C_pair)

        mask_bias = (self.inf * (pair_mask.to(torch.float) - 1))[..., None, :, :]  # (*, 1, I, J)
        pair_bias = self.linear(pair_embed) * self.linear_g(pair_embed).sigmoid()  # (*, I, J, H)
        pair_bias = permute_final_dims(pair_bias, [2, 0, 1])  # (*, H, I, J)

        node_embed_i = node_embed_i + self.dropout(self.mha(
            q_x=node_embed_i,
            kv_x=node_embed_j,
            biases=[mask_bias, pair_bias],
            distance=distance, 
        )) * node_mask_i.to(torch.float).unsqueeze(-1)

        return node_embed_i
