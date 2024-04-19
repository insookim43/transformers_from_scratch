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
    # @TODO : mask
    return attention_score

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
    def __init__(self, n_heads=None, d_model=None, d_k=None, mask = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.mask = mask

        self.WQ = nn.Linear(in_features=d_model, out_features=d_model)
        self.WK = nn.Linear(in_features=d_model, out_features=d_model)
        self.WV = nn.Linear(in_features=d_model, out_features=d_model)

        self.W_proj = nn.Linear(in_features=d_k, out_features=d_k)

    def forward(self, X):
        Q = self.WQ(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        K = self.WK(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        V = self.WV(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        attention = scaled_dot_product_attention(Q, K, V, self.mask)
        return self.W_proj(attention)

def clones(module, n=None):
    """return modulelist of as identical structure of module that is given as input argument"""
    # @TODO
    pass
    


class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=512):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0 
        d_k = self.d_model // self.n_heads
        self.multi_head_attention = MultiheadAttention(n_heads=n_heads, d_model=d_model, d_k=d_k, mask=None)
        self.layernorm_at_attention_sublayer = nn.LayerNorm([d_k])
        self.feedforward = nn.Linear(in_features=d_model, out_features=d_model) # head information is mixed
        self.layernorm_at_ff_sublayer = nn.LayerNorm([d_k])

    def forward(self, X):
        # @TODO
        attention_by_heads = self.multi_head_attention(X)
        attention_concat = attention_by_heads.view(X.shape)
        attention_plus_skip_connection = attention_concat + X
        attention_sublayer_output = self.layernorm_at_attention_sublayer(attention_plus_skip_connection)

        ff_output = self.feedforward(attention_sublayer_output)
        ff_plus_skip_connection = ff_output + attention_sublayer_output
        ff_sublayer_output = self.layernorm_at_ff_sublayer(ff_plus_skip_connection)

        return ff_sublayer_output

class TransformerEncoder(nn.Module):
    def __init__(self, n_encoder_layers=6, n_heads=8):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.encoder_layers = clones(EncoderLayer(n_heads), n_encoder_layers)
        
    def forward(self, x):
        for i in range(self.n_encoder_layers):
            x = self.encoder_layers[i](x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # @TODO
    def forward(self, x):
        # @TODO
        pass

class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # @TODO
        pass

    def forward(self, x):
        # @TODO
        pass


class ProjectionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # @TODO
        pass

    def forward(self, x):
        # @TODO
        pass

class EncoderDecoderTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TransformerEncoder()

        # @TODO
        self.decoder = TransformerDecoder()
        self.proj = ProjectionLayer()
        pass

    def forward(self, x):
        # @TODO
        pass

def make_model():
    # @TODO
    model = EncoderDecoderTransformer()

    return model

# make model
model = make_model()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

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
    train_epoch(model, train_dataloader, optimizer)

# Attention 시각화
# @TODO

# 예측
# @TODO
