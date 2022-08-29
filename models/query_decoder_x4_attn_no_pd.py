"""
Pytorch version by Song Yan
"""
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

    def forward(self, Q, K, V):

        mixed_query_layer = self.query(Q)  # [B, Q=1, 2048]
        mixed_key_layer = self.key(K)      # [B, P=49, 2048]
        mixed_value_layer = self.value(V)  # [B, Patches, C]

        query_layer = self.transpose_for_scores(mixed_query_layer)     # [B, Heads=8, Q=1, head_size=256]
        key_layer = self.transpose_for_scores(mixed_key_layer)         # [B, Heads=8, P=49, head_size=256]
        value_layer = self.transpose_for_scores(mixed_value_layer)     # [B, Heads=8, P=49, head_size=256]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B, Heads=8, Q=1, P=49]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # for each K cluster, the probality of all pxiels belong to a specific cluster centers
        attention_scores = self.softmax(attention_scores)                       # [B, Heads, Q, P]

        context_layer = torch.matmul(attention_scores, value_layer)           # [B, Head, Q, Head_size]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()      # [B, Q, heads, head_size]
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)         # [B, N_q, C=256]
        context_layer = context_layer.view(*new_shape)

        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, attention_scores # [B, N_q, C]


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size=(56, 56), patch_size=(7, 7), hidden_size=256, in_channels=256, dropout=0.1):
        super(Embeddings, self).__init__()

        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.n_patches = n_patches
        self.patch_embeddings = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = Dropout(dropout)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embeddings(x) # [B, C, H/stride, W/stride]
        x = x.flatten(2)             # [B, C, H*W=patches]
        x = x.transpose(-1, -2)      # [B, tokens, C]
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class QueryDeCoderLayer(nn.Module):
    def __init__(self, hidden_size=256, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super(QueryDeCoderLayer, self).__init__()

        # self attention
        self.self_attn = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size)

        # cross attention
        self.cross_attn = Attention(num_heads=n_heads, hidden_size=hidden_size, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_size)

        # ffn
        self.linear1 = nn.Linear(hidden_size, d_ffn)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, hidden_size)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward_ffn(self, query):
        query2 = self.linear2(self.dropout3(self.activation(self.linear1(query))))
        query = query + self.dropout4(query)
        query = self.norm3(query)
        return query

    def forward(self, query, kv=None):
        """ query : template target features
              kv  : keyvalue, pxiel features
        """
        # self attention
        q2, _ = self.self_attn(query, query, query)
        query = query + self.dropout2(q2)
        query = self.norm2(q2)

        # cross attention
        if kv is None:
            kv = query
        q2, attn = self.cross_attn(query, kv, kv)
        query = query + self.dropout1(q2)
        query = self.norm1(query)

        # ffn
        query = self.forward_ffn(query)

        return query, attn


class QueryDeCoder(nn.Module):
    """
    hidden_size: input channels
    d_ffn : dims feed forward
    """
    def __init__(self, hidden_size=2048, d_ffn=1024, dropout=0.1, activation="relu", n_heads=8):
        super(QueryDeCoder, self).__init__()

        self.decoder1 = nn.ModuleList([QueryDeCoderLayer(hidden_size, d_ffn, dropout, activation, n_heads),
                                       QueryDeCoderLayer(hidden_size, d_ffn, dropout, activation, n_heads)])

        self.decoder2 = nn.ModuleList([QueryDeCoderLayer(hidden_size//2, d_ffn//2, dropout, activation, n_heads),
                                       QueryDeCoderLayer(hidden_size//2, d_ffn//2, dropout, activation, n_heads)])

        self.decoder3 = nn.ModuleList([QueryDeCoderLayer(hidden_size//4, d_ffn//4, dropout, activation, n_heads),
                                       QueryDeCoderLayer(hidden_size//4, d_ffn//4, dropout, activation, n_heads)])

        self.decoder4 = nn.ModuleList([QueryDeCoderLayer(hidden_size//8, d_ffn//4, dropout, activation, n_heads),
                                       QueryDeCoderLayer(hidden_size//8, d_ffn//4, dropout, activation, n_heads)])

        self.downsample1 = nn.Linear(hidden_size, hidden_size//2)
        self.downsample2 = nn.Linear(hidden_size//2, hidden_size//4)
        self.downsample3 = nn.Linear(hidden_size//4, hidden_size//8)

        # FFN
        self.linear1 = nn.Linear(hidden_size//8, hidden_size//8)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_size//8, hidden_size//8)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_size//8)

    def forward(self, queries, pixel_features):
        ''' fg_queries : [B, 1, 2048], [B, 16, 1024], [B, 64, 512], [B, 144, 256]
            bg_queries : [B, 1, 2048], [B, 9, 1024], [B, 25, 512], [B, 49, 256]

        Patches: 7*7, 14*14, 14*14, 28*28
        pixel_features :  [B, Patches=49, 2048], [B, Patches=196, 1024], [B, Patches=196, 512], [B, Patches=784, 256]

        return queries, attn_maps for localization
        attn maps : [B, Heads=8, Q=1, P=49], [B, Heads=8, Q=1+16, P=49], [B, Heads=8, Q=1+16+64, P=49], [B, Heads=8, Q=1+16+64+144, P=49]
        '''
        q1, attn1 = self.decoder1[0](queries[0], pixel_features[0]) # q1 : [B, Q, C=2048], attn 1 : [B, Heads=8, Q=1, P=49]
        q1, _ = self.decoder1[1](q1)
        q1 = self.downsample1(q1)                                   # q1 : [B, Q, C=1024]

        #
        q2 = torch.cat((queries[1], q1), dim=1)                     # attn 2 : [B, Heads=8, Q=1+16, P=49]
        q2, attn2 = self.decoder2[0](q2, pixel_features[1])
        q2, _ = self.decoder2[1](q2)
        q2 = self.downsample2(q2)

        #
        q3 = torch.cat((queries[2], q2), dim=1)                      # attn 3: [B, Heads=8, Q=1+16+64, P=49]
        q3, attn3 = self.decoder3[0](q3, pixel_features[2])
        q3, _ = self.decoder3[1](q3)
        q3 = self.downsample3(q3)

        #
        q4 = torch.cat((queries[3], q3), dim=1)
        q4, attn4 = self.decoder4[0](q4, pixel_features[3]) # attn map : [B, Heads, Q=1+16+64+144, P=49] => [B, H*Q, P=49]
        q4, _ = self.decoder4[1](q4)

        # FFN
        _query = self.linear2(self.dropout3(self.activation(self.linear1(q4)))) # BxQx256, Q=(1+9+25)
        query = q4 + self.dropout4(_query)
        query = self.norm3(query)

        #
        score = F.relu(attn4)
        score = score.view(score.shape[0], -1, score.shape[-1]) # [B, HQ, P]
        score = torch.mean(score, dim=1) # [B, P]

        return query, score

if __name__ == '__main__':

    model = QueryDeCoder()
    query = [torch.zeros(32, 1, 2048), torch.zeros(32, 16, 1024), torch.zeros(32, 64, 512), torch.zeros(32, 256, 256)]
    pixel_features = [torch.zeros(32, 49, 2048), torch.zeros(32, 49, 1024), torch.zeros(32, 49, 512), torch.zeros(32, 49, 256)]
    output, score = model(query, pixel_features)
    print(output.shape) # [32, 10, 256], perform mask head on [32, 256, 56, 56]
