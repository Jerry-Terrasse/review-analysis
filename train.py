import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn import functional as F

import glob
import time
import json
import pickle

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
        
        tokens = list(map(lambda x: self.glove.word2idx.get(x, 0), quote))
        tokens_ = torch.tensor(tokens)
        self.cache[idx] = tokens_
        
        return tokens_, score_

class Net(nn.Module):
    def __init__(self, pre_trained: list[torch.Tensor], num_hidden=128, num_layers=2):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        
        embed_matrix = torch.stack(pre_trained, dim=0)
        self.embedding = nn.Embedding.from_pretrained(embed_matrix)
        self.lstm = nn.LSTM(embed_matrix.shape[-1], num_hidden, num_layers)
        
        self.output = nn.Sequential(
            nn.Linear(num_hidden, 1),
            # nn.Tanh(),
            # nn.Linear(30, 1),
            nn.Sigmoid()
        )
    
    def forward(self, inputs, state):
        X = self.embedding(inputs.T)
        Y, state = self.lstm(X, state)
        output = self.output(Y)
        return output, state
    
    def begin_state(self, device, batch_size=1):
        # return torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        return (
            torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device),
            torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        )

def grad_clipping(net, theta):
    params = [p for p in net.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    logger.debug(f'grad norm: {norm}')
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm

def evaluate(net, val_loader, loss, device):
    net.eval()
    losses = []
    results = []
    predicts = []
    realitys = []
    for X, Y in val_loader:
        state = net.begin_state(device, batch_size=X.shape[0])
        X, Y = X.to(device), Y.to(device)
        y_hat, state = net(X, state)
        final_y_hat = y_hat[-1].reshape(-1)
        
        l = loss(Y, final_y_hat)
        losses.append(l.item())
        
        predict = final_y_hat.item()
        reality = Y.item()
        results.append((predict, reality))
        predicts.append(predict)
        realitys.append(reality)
    logger.info(f'Predict: {predicts[: 4]}')
    logger.info(f'Reality: {realitys[: 4]}')
    return sum(losses) / len(losses)

def train(net, train_loader, val_loader, lr, num_epochs, device='cuda:0'):
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    val_loss = evaluate(net, val_loader, loss, device)
    logger.info(f'Initial val loss: {val_loss:.6f}')
    
    loss_list: list[float] = []
    val_loss_list: list[float] = []
    
    for epoch in tqdm(range(1, num_epochs+1)):
        net.train()
        l_sum = torch.tensor(0., device=device)
        for X, Y in train_loader:
            state = net.begin_state(device, batch_size=X.shape[0])
            
            X, Y = X.to(device), Y.to(device)
            y_hat, state = net(X, state)
            final_y_hat = y_hat[-1].reshape(-1)
            # import pdb; pdb.set_trace()
            # Y = Y.repeat(y_hat.shape[0], 1).reshape(-1, 1, 1)
            l = loss(Y, final_y_hat)
            # logger.debug(Y)
            # logger.debug(f"Y {Y.norm()} y_hat {y_hat.norm()} loss {l.item()}")
            
            l_sum += l
            # logger.info([x.grad for x in optimizer.param_groups[0]['params']])
            # input("pause")
        
        optimizer.zero_grad()
        l_sum.backward()
        grad_clipping(net, 10.)
        optimizer.step()
        
        logger.info(f'Epoch {epoch} loss: {l_sum.item() / len(train_loader):.6f}')
        val_loss = evaluate(net, val_loader, loss, device)
        logger.info(f'Epoch {epoch} val loss: {val_loss:.6f}')
        
        loss_list.append(l_sum.item() / len(train_loader))
        val_loss_list.append(val_loss)
        plot_loss(loss_list, 'loss_img/loss.png')
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'./ckpts/model_{epoch}.pt')

@logger.catch(reraise=True)
def main():
    dataset_path = './theater_eng.json'
    dataset = TomatoDataset(dataset_path)
    if len(dataset) < 100:
        logger.warning(f"Too few data: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    # Weighted Sample
    # TODO
    
    train_loader, test_loader = DataLoader(train_data, batch_size=1), DataLoader(test_data, batch_size=1)

    logger.info(f'Train size: {len(train_loader)}')
    logger.info(f'Test size: {len(test_loader)}')

    # logger.debug(next(iter(train_loader))[0])

    num_hidden, num_layers = 128, 1
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
    
    train(net, train_loader, test_loader, 0.02, 100, device)


if __name__ == "__main__":
    main()