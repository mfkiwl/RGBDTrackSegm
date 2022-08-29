import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from collections import OrderedDict

''' Code from
    An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
    https://github.com/jeonsworld/ViT-pytorch/blob/main/models/modeling.py
'''

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=(56, 56), patch_size=(7, 7), out_dim=256, in_dim=256, dropout=0.1):
        super(Embeddings, self).__init__()

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.n_patches = n_patches
        self.patch_embeddings = nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, out_dim))
        self.dropout = Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embeddings(x) # [B, C, H/stride, W/stride]
        x = x.flatten(2)             # [B, C, H*W=patches]
        x = x.transpose(-1, -2)      # [B, tokens, C]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module):
    def __init__(self, num_heads=8, hidden_size=256, dropout=0.1):
        super(Attention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / self.num_attention_heads)   # 32
        self.all_head_size = self.num_attention_heads * self.attention_head_size # 8 heads * 32

        self.query = nn.Linear(hidden_size, self.all_head_size) # 256 -> 256
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.softmax = nn.Softmax(dim=-1) # along Pixel dim

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)     # [B, Heads, Patches, Head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B,Heads, Patches, Patches]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.softmax(attention_scores)
        weights = attention_scores

        context_layer = torch.matmul(attention_scores, value_layer)           # [B, Head, Patches, Head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()        # [B, Patches, heads, head_size]
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)         # [B, Patches, C=256]
        context_layer = context_layer.view(*new_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights # [B, Patches, C]

class Mlp(nn.Module):
    def __init__(self, hidden_size, mlp_dim, dropout=0.1):
        super(Mlp, self).__init__()
        self.fc1 = Linear(hidden_size, mlp_dim)
        self.fc2 = Linear(mlp_dim, hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, hidden_size, mlp_size, dropout=0.1):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attention_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(hidden_size, eps=1e-6)
        self.ffn = Mlp(hidden_size, mlp_size, dropout=dropout)
        self.attn = Attention(num_heads=8, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Encoder(nn.Module):
    def __init__(self, num_layers=6, hidden_size=256, mlp_size=256, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_size, dropout=dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Decoder(nn.Module):
    def __init__(self, num_layers=6, hidden_size=256, mlp_size=256, dropout=0.1):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(hidden_size, eps=1e-6)
        for _ in range(num_layers):
            layer = Block(hidden_size, mlp_size, dropout=dropout)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, query, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights
