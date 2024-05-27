# import libraries
from transformers import BertTokenizer, BertModel
from transformers import PreTrainedTokenizerFast
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
import time


# 변수 정의
num_epochs = 2
batch_size = 2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.00001
# model parameters
d_model = 512
d_ff = 2048
embedding_option = 'bert'
if embedding_option == 'bert':
    d_model = 768 # bert embedding is 768 dim
embedding_option_all = ['scratch', 'bert']
emb_size = 768
dropout = 0.1
maxlen = 512 # max seq length

class ToyDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class TokenDataset(Dataset):

    def __init__(self, x, y, src_tokenizer, tgt_tokenizer, embedding):
        """x : batched raw src txt, 
        y : batched raw tgt txt,
        self.x_encode : encoded objects of src 
        self.y_encode : encoded objects of tgt
        src_tokenizer : transformers tokenizer encoding object
        tgt_tokenizer : transformers tokenizer encoding object"""
        self.x = x
        self.y = y
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        
        self.x_encode = self.src_tokenizer(
            self.x,
            return_tensors='pt',     # 텐서로 반환
            truncation=False,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
)

        self.y_encode = self.tgt_tokenizer(
            self.y,                 
            return_tensors='pt',     # 텐서로 반환
            truncation=False,         # 잘라내기 적용
            padding='max_length',    # 패딩 적용
            add_special_tokens=True  # 스페셜 토큰 적용
        )
        self.x_ids = self.x_encode['input_ids']
        self.y_ids = self.y_encode['input_ids']

        self.x_ids_unsqueezed = self.x_ids.unsqueeze(-1)
        self.y_ids_unsqueezed = self.y_ids.unsqueeze(-1)

        # self.x_ids_embedded = embedding(self.x_ids)
        # self.y_ids_embedded = embedding(self.y_ids)

    def generate_tgt_mask(self, seq_length):
        # @TODO: mask [PAD] token 
        mask = torch.triu(torch.ones((seq_length, seq_length),
                          dtype=torch.float), diagonal=1)
        return mask

    def __len__(self):
        return len(self.x)

    # def __getitem__(self, idx):
        
    #     x_tokens = self.x_ids_embedded[idx].to(device)
    #     y_tokens = self.y_ids_embedded[idx].to(device)
    #     seq_length = y_tokens.shape[-2]
    #     tgt_mask = self.generate_tgt_mask(seq_length=seq_length).to(device)

    #     return x_tokens, y_tokens, tgt_mask

    def __getitem__(self, idx):
        
        x_id = self.x_ids_unsqueezed[idx].to(device)
        y_id = self.y_ids_unsqueezed[idx].to(device)
        seq_length = y_id.shape[-2]
        tgt_mask = self.generate_tgt_mask(seq_length=seq_length).to(device)

        return x_id, y_id, tgt_mask


class embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d_model)

    def forward(self, X):
        return self.embedding(X)

# encoding

# @TODO : class로 wrap, 최종 output에 dropout, batch 입력에 대응 
from torch import Tensor
import math
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size) # 왜 exp 하지?
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout) # 왜 dropout 하지?!
        self.register_buffer('pos_embedding', pos_embedding) # 이건 또 머야?

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding)

def apply_mask(orig, mask):
    masked_orig = orig.masked_fill(mask==1, 1e-9)
    return masked_orig


def scaled_dot_product_attention(Q, K, V, mask):
    scale = K.shape[-1] ** 0.5
    QK_T = torch.matmul(Q, torch.transpose(K, -2, -1))
    if mask is not None:
        QK_T = apply_mask(QK_T, mask.unsqueeze(-3))   
    attention_score = torch.softmax(QK_T, dim=-1) / scale
    out = torch.matmul(attention_score, V)
    return out


def clones(module, n=None):
    """return modulelist of as identical structure of module instance given as input argument"""
    return nn.ModuleList([module for _ in range(n)])


class MultiheadAttention(nn.Module):
    def __init__(self, n_heads=None, d_model=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        self.WQ = nn.Linear(in_features=d_model, out_features=d_model)
        self.WK = nn.Linear(in_features=d_model, out_features=d_model)
        self.WV = nn.Linear(in_features=d_model, out_features=d_model)
        self.W_proj = nn.Linear(in_features=d_model, out_features=d_model)

    def forward(self, target, source, memory, mask=None):
        batch_size = target.shape[0]
        d_k = self.d_model // self.n_heads
        Q = self.WQ(target).view([batch_size, self.n_heads, -1, d_k])
        K = self.WK(source).view([batch_size, self.n_heads, -1, d_k])
        V = self.WV(memory).view([batch_size, self.n_heads, -1, d_k])
        attention = scaled_dot_product_attention(Q, K, V, mask)
        attention = attention.transpose(1, 2).contiguous() \
            .view(target.shape[0], -1, self.d_model)
        return self.W_proj(attention)


class EncoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=None, d_ff=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0
        self.d_ff = d_ff
        self.multi_head_attention = MultiheadAttention(
            n_heads=n_heads, d_model=d_model)
        self.layernorm_at_attention_sublayer = nn.LayerNorm(d_model)
        # head information is mixed
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=self.d_ff),
            nn.Linear(in_features=self.d_ff, out_features=d_model))
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
    def __init__(self, n_encoder_layers=6, n_heads=8, d_model=None, d_ff=None):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.encoder_layers = clones(
            EncoderLayer(n_heads=n_heads, d_model=d_model, d_ff=d_ff), self.n_encoder_layers)

    def forward(self, X):
        for i in range(self.n_encoder_layers):
            X = self.encoder_layers[i](X)
        return X


class DecoderLayer(nn.Module):
    def __init__(self, n_heads=8, d_model=None, d_ff=None):
        super().__init__()
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.d_model = d_model
        assert self.d_model % self.n_heads == 0
        self.multi_head_attention = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model)
        self.layernorm_at_self_attention_sublayer = nn.LayerNorm(d_model)
        self.cross_attention = MultiheadAttention(
            n_heads=n_heads,
            d_model=d_model
        )
        self.layernorm_at_cross_attention_sublayer = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(in_features=d_model, out_features=self.d_ff),
            nn.Linear(in_features=self.d_ff, out_features=d_model))
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
        cross_attention_concat = cross_attention_by_heads.view(
            X.shape)
        cross_attention_plus_skip_connection = cross_attention_concat + \
            self_attention_sublayer_output
        cross_attention_sublayer_output = self.layernorm_at_cross_attention_sublayer(
            cross_attention_plus_skip_connection)

        ff_output = self.feedforward(cross_attention_sublayer_output)
        ff_plus_skip_connection = ff_output + cross_attention_sublayer_output
        ff_sublayer_output = self.layernorm_at_ff_sublayer(
            ff_plus_skip_connection)

        return ff_sublayer_output


class TransformerDecoder(nn.Module):
    def __init__(self, n_decoder_layers=6, n_heads=8, d_model=None):
        super().__init__()
        self.n_decoder_layers = n_decoder_layers
        self.decoder_layers = clones(
            DecoderLayer(n_heads=n_heads, d_model=d_model, d_ff=d_ff), self.n_decoder_layers)

    def forward(self, X, memory, tgt_mask):

        for i in range(self.n_decoder_layers):
            X = self.decoder_layers[i](X, memory, memory, tgt_mask)
        return X


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.feedforward = nn.Linear(in_features=d_model, out_features=vocab)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, X):
        ff_out = self.feedforward(X)
        # @TODO
        return ff_out


class EncoderDecoderTransformer(nn.Module):
    def __init__(self, d_model, d_vocab, d_ff, src_vocab_size, tgt_vocab_size, emb_size, dropout, maxlen):
        super().__init__()
        self.encoder = TransformerEncoder(n_encoder_layers=6, n_heads=8, d_model=d_model, d_ff=d_ff)
        self.d_model = d_model
        self.decoder = TransformerDecoder(n_decoder_layers=6, n_heads=8, d_model=d_model)
        self.proj = ProjectionLayer(self.d_model, d_vocab)
        self.src_tok_emb = embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout, maxlen)

    def forward(self, source, target, tgt_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(source))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(target))
        encoder_out = self.encoder(src_emb)
        out = self.decoder(tgt_emb, encoder_out, tgt_mask)
        logit = self.proj(out)        
        return F.log_softmax(logit, dim=-1)


def make_model(d_model, d_vocab, d_ff, src_vocab_size, tgt_vocab_size, emb_size, dropout, maxlen):
    model = EncoderDecoderTransformer(d_model=d_model, 
                                      d_vocab=d_vocab,
                                      d_ff=d_ff,
                                      src_vocab_size=src_vocab_size, 
                                      tgt_vocab_size=tgt_vocab_size, 
                                      emb_size=emb_size,
                                      dropout=dropout,
                                      maxlen=maxlen)
    print(model)

    # init model param
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model



# 데이터
# 4*3*2 toy data
toy_random_data_raw = torch.rand(2, 5, 512)
toy_random_target_raw = torch.rand(2, 5, 512)
toy_random_train_dataset = ToyDataset(
    toy_random_data_raw, toy_random_target_raw)
train_dataloader = DataLoader(toy_random_train_dataset, batch_size=batch_size)
print(next(iter(train_dataloader))[0].shape)


toy_data_raw = ["My favorite fruit is orange.",
                "No one likes orange."]
toy_target_raw = ["제가 제일 좋아하는 과일은 오렌지입니다.",
                  "오렌지를 좋아하는 사람은 없습니다."]

src_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tgt_tokenizer = PreTrainedTokenizerFast.from_pretrained("/app/transformer/embedding/BPE-nsmc.model")

# len of embedding 
print("len of src_tokenizer: ",  len(src_tokenizer))
print("len of tgt_tokenizer: ", len(tgt_tokenizer))

toy_data = src_tokenizer(toy_data_raw)
toy_target = tgt_tokenizer(toy_target_raw)

print([src_tokenizer.convert_ids_to_tokens(encoded) 
       for encoded in toy_data['input_ids']])
print([tgt_tokenizer.convert_ids_to_tokens(encoded) 
       for encoded in toy_target['input_ids']])
print("print ids of toy src & tgt")
print([encoded for encoded in toy_data['input_ids']])
print([encoded for encoded in toy_target['input_ids']])

if embedding_option in embedding_option_all:
    if embedding_option == 'scratch':
        embedding_layer = embedding(len(src_tokenizer), d_model=d_model) # src tokenizer
    elif embedding_option == 'bert':
        model = BertModel.from_pretrained("bert-base-uncased") # embedding dim : 768
        embedding_layer = model.embeddings.word_embeddings

toy_train_dataset = TokenDataset(toy_data_raw,
                             toy_target_raw,
                             src_tokenizer,
                             tgt_tokenizer,
                             embedding_layer)

train_dataloader = DataLoader(toy_train_dataset, batch_size=batch_size)
print("first encoded ids in dataloader")
print(next(iter(train_dataloader)))


# make model
model = make_model(d_model=d_model, 
                   d_vocab=len(tgt_tokenizer), 
                   d_ff=d_ff, 
                   src_vocab_size=len(src_tokenizer), 
                   tgt_vocab_size=len(tgt_tokenizer), 
                   emb_size=emb_size, 
                   dropout=dropout,
                   maxlen=maxlen)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
optimizer.zero_grad()

# lr Scheduler
scheduler = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1, end_factor=0.1)

# loss criterion 정의


class LabelSmoothing(nn.Module):
    """implement label smoothing. """

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()

        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        # except gt and pad token
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)

        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, true_dist)


loss_fn = torch.nn.KLDivLoss()

# train 함수 정의


def train_epoch(model, train_dataloader, optimizer, loss_fn, scheduler):
    model.train()
    losses = 0
    for src, tgt, tgt_mask in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        optimizer.zero_grad()
        logits = model(src, tgt, tgt_mask)
        print("logits", logits)
        loss = loss_fn(logits, tgt) # @TODO : label smoothing tgt
        loss.backward()

        optimizer.step()
        losses += loss.item()
        # scheduler.step()

    return losses / len(list(train_dataloader))

# evaluate 함수 정의


def evaluate(model, val_dataloader, loss_fn):
    model.eval()
    losses = 0
    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        logits = model(src, tgt)

        # loss = loss_fn
        # losses += loss.item()

    # return losses / len(list(val_dataloader))


# 훈련
model.to(device)

for epoch in range(num_epochs):
    train_epoch(model, train_dataloader, optimizer, loss_fn, scheduler)
    evaluate()


# evaluate

# Attention 시각화
# @TODO

# 예측
# @TODO
