# import libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim

import numpy as np
# @TODO

# 변수 정의
num_epochs = 2
batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.00001


# 데이터
toy_data_raw = torch.Tensor(np.array([
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12]],
    [[13, 14], [15, 16], [17, 18]],
    [[19, 20], [21, 22], [23, 24]]]))
toy_target_raw = torch.Tensor(np.array([
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12]],
    [[13, 14], [15, 16], [17, 18]],
    [[19, 20], [21, 22], [23, 24]]]))


class ToyDataset(Dataset):
    def __init__(self, x, y):
        # @TODO
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        # @TODO
        pass


toy_sample = ToyDataset(toy_data_raw, toy_target_raw)
train_dataloader = DataLoader(toy_sample, batch_size=batch_size)


def apply_mask(attention_score, mask):
    masked_attention_score = attention_score + mask[None, None, ...] # b h s s
    return masked_attention_score


def scaled_dot_product_attention(Q, K, V, mask):
    scale = K.shape[-1] ** 0.5
    attention_score = torch.softmax(torch.matmul(
        Q, K.permute([*torch.arange(Q.ndim)-2, -1, -2])), dim=-1)
    if mask is not None:
        attention_score = apply_mask(attention_score, mask)
    out = torch.matmul(attention_score, V)
    out = torch.reciprocal(scale) * out
    return out


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads=None, d_model=None, d_k=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.WQ = nn.Linear(in_features=d_model, out_features=d_model)
        self.WK = nn.Linear(in_features=d_model, out_features=d_model)
        self.WV = nn.Linear(in_features=d_model, out_features=d_model)

        self.W_proj = nn.Linear(in_features=d_k, out_features=d_k)

    def forward(self, target, source, memory, mask=None):
        if mask:
            self.mask = mask
        Q = self.WQ(target).view([*target.shape[:-1]+[self.n_heads, -1]])
        K = self.WK(source).view([*source.shape[:-1]+[self.n_heads, -1]])
        V = self.WV(memory).view([*memory.shape[:-1]+[self.n_heads, -1]])
        attention = scaled_dot_product_attention(Q, K, V, self.mask)
        return self.W_proj(attention)


def clones(module, n=None):
    """return modulelist of as identical structure of module instance given as input argument"""
    return nn.ModuleList([module for _ in range(n)])

class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0
        d_k = self.d_model // self.n_heads
        self.multi_head_attention = MultiheadAttention(
            n_heads=n_heads, d_model=d_model, d_k=d_k)
        self.layernorm_at_attention_sublayer = nn.LayerNorm(d_model)
        # head information is mixed
        self.feedforward = nn.Linear(in_features=d_model, out_features=d_model)
        self.layernorm_at_ff_sublayer = nn.LayerNorm(d_model)

    def forward(self, X):
        attention_by_heads = self.multi_head_attention(X, X, X, mask=None)
        attention_concat = attention_by_heads.view(X.shape)
        attention_plus_skip_connection = attention_concat + X
        attention_sublayer_output = self.layernorm_at_attention_sublayer(
            attention_plus_skip_connection)

        ff_output = self.feedforward(attention_sublayer_output)
        ff_plus_skip_connection = ff_output + attention_sublayer_output
        ff_sublayer_output = self.layernorm_at_ff_sublayer(
            ff_plus_skip_connection)

        return ff_sublayer_output


class TransformerEncoder(nn.Module):
    def __init__(self, n_encoder_layers=6, n_heads=8):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.encoder_layers = clones(EncoderLayer(n_heads), self.n_encoder_layers)

    def forward(self, X):
        for i in range(self.n_encoder_layers):
            X = self.encoder_layers[i](X)
        return X


class DecoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0
        d_k = self.d_model // self.n_heads
        self.multi_head_attention = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_k)
        self.layernorm_at_self_attention_sublayer = nn.LayerNorm(d_k)
        self.cross_attention = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model,
            d_k=d_k
        )
        self.layernorm_at_cross_attention_sublayer= nn.LayerNorm(d_model)
        self.feedforward = nn.Linear(in_features=d_model, out_features=d_model)
        self.layernorm_at_ff_sublayer = nn.LayerNorm(d_model)

    def forward(self, X, source, memory, mask):
        self_attention_by_heads = self.multi_head_attention(X, X, X, mask=mask)
        self_attention_concat = self_attention_by_heads.view(X.shape)
        self_attention_plus_skip_connection = self_attention_concat + X
        self_attention_sublayer_output = self.layernorm_at_self_attention_sublayer(
            self_attention_plus_skip_connection)
        
        cross_attention_by_heads = self.cross_attention(self_attention_sublayer_output, 
                                                        source, 
                                                        memory, 
                                                        mask=None)
        cross_attention_concat = cross_attention_by_heads.view(cross_attention_by_heads.shape)
        cross_attention_plus_skip_connection = cross_attention_concat + self_attention_sublayer_output
        cross_attention_sublayer_output = self.layernorm_at_cross_attention_sublayer(
            cross_attention_plus_skip_connection) 
        
        ff_output = self.feedforward(cross_attention_sublayer_output)
        ff_plus_skip_connection = ff_output + cross_attention_sublayer_output
        ff_sublayer_output = self.layernorm_at_ff_sublayer(
            ff_plus_skip_connection)
        
        return ff_sublayer_output


class TransformerDecoder(nn.Module):
    def __init__(self, n_decoder_layers=6, n_heads=8):
        super().__init__()
        self.n_decoder_layers = n_decoder_layers
        self.decoder_layers = clones(DecoderLayer(n_heads), self.n_decoder_layers)

    def generate_mask(self, seq_length):
        mask = torch.triu(torch.ones((seq_length, seq_length), dtype=torch.float), diagonal=1)
        mask = mask * (-1e+9)
        return mask

    def forward(self, X, memory):
        seq_length = X.shape[1] # b "s" d_model
        mask = self.generate_mask(seq_length)

        for i in range(self.n_decoder_layers):
            X = self.decoder_layers[i](X, memory, mask)
        return X


class ProjectionLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.feedforward = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X):
        ff_out = self.feedforward(X)
        return torch.nn.Softmax(ff_out)


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.encoder = TransformerEncoder()
        self.d_model = d_model
        self.decoder = TransformerDecoder()
        self.proj = ProjectionLayer(self.d_model)
        # @TODO
        pass

    def forward(self, x):
        # @TODO
        pass


def make_model():
    # @TODO
    model = EncoderDecoderTransformer(d_model=512)
    print(model)
    return model


# make model
model = make_model()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.1)
# lr Scheduler
# loss criterion 정의
# train 함수 정의
model.to(device)
model.train()


def train_epoch(model, dataloader, optimizer, scheduler):
    # @TODO
    pass
    # Data embedding
    # @TODO
    # Data encoding
    # @TODO


for epoch in range(num_epochs):
    train_epoch(model, train_dataloader, optimizer, scheduler)

# Attention 시각화
# @TODO

# 예측
# @TODO
