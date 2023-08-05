import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F

import glob
import time
import json
import pickle
import math
import random

from typing import Callable
from collections import Counter

from loguru import logger
from tqdm import tqdm
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

from utils import Glove, plot_loss, plot_hist

class Review:
    def __init__(self, quote: str, score: float) -> None:
        self.quote = quote.strip()
        self.score = score
        self.isacii = self.quote.isascii()
    def __repr__(self) -> str:
        return f'''Rating: {self.score}
{self.review}
'''

def cut_or_pad(quote: list[str], length: int) -> list[str]:
    if len(quote) > length:
        return quote[:length]
    else:
        return quote + ['<pad>'] * (length - len(quote))

class TomatoDataset(Dataset):
    def __init__(self, data_file: str):
        data = json.load(open(data_file, 'r'))
        
        # to balance the dataset
        scores = list(map(lambda x: x[0], data))
        counter = Counter(scores)
        p_list = np.array([1 / counter[score] for score in counter])
        score2idx = {score: idx for idx, score in enumerate(counter)}
        p_list /= p_list.sum()
        data, data_ = [], data
        for _ in range(10):
            for item in data_:
                if np.random.rand() < p_list[score2idx[item[0]]]:
                    data.append(item)
        
        self.scores: list[float] = list(map(lambda x: x[0], data))
        self.quotes: list[list[str]] = list(map(lambda x: x[1].split(), data))
        plot_hist([len(x) for x in self.quotes], bins=range(0, 600, 20), fname='hist.png')
        
        counter = Counter(self.scores)
        logger.info(f'Counter: {counter}')
        
        self.cache: dict[int, torch.Tensor] = {}
        logger.info(f'Loaded {len(self.quotes)} reviews')
        
        logger.info(f'Loading Glove...')
        glove_path = './pretrain/glove.6B.100d.txt'
        glove_cache = glove_path + '.cache.json'
        if glob.glob(glove_cache):
            logger.info(f'Loading Glove from cache')
            self.glove = Glove(glove_cache, 100, cache=True)
        else:
            self.glove = Glove(glove_path, 100)
            self.glove.dump(glove_cache)
        logger.info(f'Loaded Glove')
        
        self.words = Counter()
        for quote in self.quotes:
            self.words.update(quote)
        
        for word, cnt in self.words.items():
            if word not in self.glove.word2idx and cnt > 5:
                logger.warning(f'Word {word} not in glove, which appears {cnt} times')

    def __len__(self) -> int:
        return len(self.scores)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        score_ = torch.tensor(self.scores[idx] / 5)
        if idx in self.cache:
            return self.cache[idx], score_
        
        quote = self.quotes[idx]
        quote = cut_or_pad(quote, 100)
        
        tokens = list(map(lambda x: self.glove.word2idx.get(x, 0), quote))
        tokens_ = torch.tensor(tokens)
        self.cache[idx] = tokens_
        
        return tokens_, score_

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Defined in :numref:`sec_seq2seq_decoder`"""
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

#@save
def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    # X:3D张量，valid_lens:1D或2D张量
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

class Attention(nn.Module):
    def __init__(self, o_size: int, q_size: int, k_size: int, v_size: int, dropout: float):
        super().__init__()
        self.W_q = nn.Linear(q_size, o_size, bias=False)
        self.W_k = nn.Linear(k_size, o_size, bias=False)
        self.W_v = nn.Linear(v_size, o_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        values = self.W_v(values)
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)

class Encoder(nn.Module):
    def __init__(self, pre_trained: list[torch.Tensor], num_hiddens, num_layers, dropout=0):
        super().__init__()
        embed_matrix = torch.stack(pre_trained, dim=0)
        self.embedding = nn.Embedding.from_pretrained(embed_matrix)
        self.rnn = nn.GRU(embed_matrix.shape[-1], num_hiddens, num_layers, dropout=dropout)
    
    def forward(self, X):
        # X: (B, S)
        X = self.embedding(X) # (B, S, E)
        X = X.permute(1, 0, 2) # (S, B, E)
        output, state = self.rnn(X)
        # output: (S, B, H)
        # state: (L, B, H)
        return output, state

class Net(nn.Module):
    def __init__(self, pre_trained: list[torch.Tensor], num_hiddens=128, num_layers=2, dropout=0.):
        super().__init__()
        self.encoder = Encoder(pre_trained, num_hiddens, num_layers, dropout)
        self.attention = Attention(num_hiddens, num_hiddens, num_hiddens, num_hiddens, dropout)
        self.output = nn.Sequential(
            nn.Linear(num_hiddens, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, X):
        enc_output, state = self.encoder(X)
        # enc_output: (S, B, H)
        # state: (L, B, H)
        query = state[-1].unsqueeze(1) # (B, 1, H)
        enc_output = enc_output.permute(1, 0, 2) # (B, S, H)
        output = self.attention(query, enc_output, enc_output)
        # output: (B, 1, H)
        Y = self.output(output.squeeze(1))
        return Y
        
        

# class Net(nn.Module):
#     def __init__(self, pre_trained: list[torch.Tensor], num_hidden=128, num_layers=2):
#         super().__init__()
#         self.num_hidden = num_hidden
#         self.num_layers = num_layers
        
#         embed_matrix = torch.stack(pre_trained, dim=0)
#         self.embedding = nn.Embedding.from_pretrained(embed_matrix)
#         self.lstm = nn.LSTM(embed_matrix.shape[-1], num_hidden, num_layers)
        
#         self.output = nn.Sequential(
#             nn.Linear(num_hidden, 1),
#             # nn.Tanh(),
#             # nn.Linear(30, 1),
#             nn.Sigmoid()
#         )
    
#     def forward(self, inputs, state):
#         X = self.embedding(inputs.T)
#         Y, state = self.lstm(X, state)
#         output = self.output(Y)
#         return output, state
    
#     def begin_state(self, device, batch_size=1):
#         # return torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
#         return (
#             torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device),
#             torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
#         )

def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if random.random() < 0.01:
        logger.debug(f'grad norm: {norm}')
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def evaluate(net, val_loader, loss, device):
    net.eval()
    losses = []
    results = []
    predicts, pred_level = [], []
    realitys, real_level = [], []
    def discriminator(x):
        return round(x * 10)
    for X, Y in val_loader:
        X, Y = X.to(device), Y.to(device)
        y_hat = net(X)
        
        l = loss(Y, y_hat.reshape(-1))
        losses.append(l.mean().item())
        
        predict = [x.item() for x in y_hat]
        reality = [x.item() for x in Y]
        results.extend(list(zip(predict, reality)))
        predicts.extend(predict)
        realitys.extend(reality)
        pred_level.extend([discriminator(x) for x in predict])
        real_level.extend([discriminator(x) for x in reality])
    logger.info(f'Predict: {predicts[: 4]}')
    logger.info(f'Reality: {realitys[: 4]}')
    correct = sum([1 if x == y else 0 for x, y in zip(pred_level, real_level)])
    correct2 = sum([1 if abs(x - y) <= 1 else 0 for x, y in zip(pred_level, real_level)])
    acc = correct / len(pred_level)
    acc2 = correct2 / len(pred_level)
    return sum(losses) / len(losses), [acc, acc2]

def train(net, train_loader, val_loader, lr, num_epochs, device='cuda:0'):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    val_loss, acc = evaluate(net, val_loader, loss, device)
    logger.info(f'Initial val loss: {val_loss:.6f}')
    
    loss_list: list[float] = []
    val_loss_list: list[float] = []
    
    for epoch in tqdm(range(1, num_epochs+1)):
        net.train()
        losses = []
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            y_hat = net(X)
            l = loss(Y, y_hat.reshape(-1))
            # logger.debug(Y)
            # logger.debug(f"Y {Y.norm()} y_hat {y_hat.norm()} loss {l.item()}")
            
            optimizer.zero_grad()
            l.mean().backward()
            grad_clipping(net, 10.)
            optimizer.step()
            # logger.info([x.grad for x in optimizer.param_groups[0]['params']])
            # input("pause")
            losses.append(l.mean().item())
        
        logger.info(f'Epoch {epoch} loss: {sum(losses) / len(losses):.6f}')
        val_loss, val_acc = evaluate(net, val_loader, loss, device)
        logger.info(f'Epoch {epoch} val loss: {val_loss:.6f}')
        logger.success(f'Epoch {epoch} val acc: {val_acc[0]*100:.2f}% {val_acc[1]*100:.2f}%')
        
        loss_list.append(sum(losses) / len(losses))
        val_loss_list.append(val_loss)
        plot_loss(loss_list, 'loss_img/loss.png')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'./ckpts/model_{epoch}.pt')

@logger.catch(reraise=True)
def main():
    dataset_path = './home_eng_clip.json'
    # dataset_path = './theater_eng_clip.json'
    dataset = TomatoDataset(dataset_path)
    if len(dataset) < 100:
        logger.warning(f"Too few data: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Weighted Sample
    # TODO
    
    train_loader, test_loader = DataLoader(train_data, batch_size=128), DataLoader(test_data, batch_size=128)

    logger.info(f'Train size: {len(train_loader)}')
    logger.info(f'Test size: {len(test_loader)}')

    # logger.debug(next(iter(train_loader))[0])

    num_hidden, num_layers = 128, 4
    device = 'cuda:0'

    net = Net(dataset.glove.idx2vec, num_hidden, num_layers)

    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
        elif isinstance(m, nn.RNN):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
        # elif isinstance(m, nn.Embedding):
        #     nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, Net):
            pass
        else:
            logger.warning(f'Not initialized: {m.__class__.__name__}')

    net.to(device)
    
    train(net, train_loader, test_loader, 0.001, 300, device)


if __name__ == "__main__":
    main()