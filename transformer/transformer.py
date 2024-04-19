# import libraries
import torch
import torch.nn as nn
import torch.utils.data.dataloader as Dataloader
import torch.utils.data.dataset as Dataset
import torch.optim.adam as Adam
# @TODO

import numpy as np

# 변수 정의
num_epochs = 2
batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.00001


# 데이터
data = np.array([
    [[1, 2], [3, 4], [5, 6]],
    [[7, 8], [9, 10], [11, 12]],
    [[13, 14], [15, 16], [17, 18]],
    [[19, 20], [21, 22], [23, 24]]])

class ToyDataset(Dataset):
    def __init__(self, dat, tar):
        super().__init__(dat, tar)
        # @TODO
        pass

    def __getitem__(self, index):
        # @TODO
        pass


train_dataloader = Dataloader(data, batch_size=batch_size)


def apply_mask(attention_score, mask):
    # @TODO : mask
    return attention_score

def dot_product_attention(Q, K, V, mask):
    scale = K.ndim ** 0.5
    attention_score = torch.softmax(torch.matmul(
        Q, K.permute([*torch.arange(Q.ndim)-2, -1, -2])), dim=-1)
    if mask is not None:
        attention_score = apply_mask(attention_score, mask)
    out = torch.matmul(attention_score, V)
    out = torch.reciprocal(scale) * out
    return out

class multi_head_attention(nn.Module):
    def __init__(self, n_heads=8, d_model = 512, mask = None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert self.d_model % self.n_heads == 0 
        d_k = self.d_model // self.n_heads
        self.mask = mask

        self.WQ = nn.Linear(in_features=d_model, out_features=d_model)
        self.WK = nn.Linear(in_features=d_model, out_features=d_model)
        self.WV = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_proj = nn.Linear(in_features=d_k, out_features=d_k)

    def forward(self, X, mask):
        Q = self.WQ(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        K = self.WK(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        V = self.WV(X).view([*X.shape[:-1]+[self.n_heads, -1]])
        attention = dot_product_attention(Q, K, V, mask)
        out = attention.W_proj(attention).view(X.shape)
        return out

def clones(module, n=None):
    """return modulelist of as identical structure of module that is given as input argument"""   
    # @TODO
    pass


class EncoderLayer(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads
        # @TODO
        pass

    def forward(self, x):
        # @TODO
        pass

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
        # @TODO
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
optimizer = Adam(model.parameters(), lr=lr)

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
