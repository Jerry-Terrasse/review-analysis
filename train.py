import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

import glob
import time
import json

from typing import Callable
from collections import Counter

from loguru import logger
from tqdm import tqdm
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)


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
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.files = glob.glob(root_dir + '/*.json')
        self.data: list[Review] = []
        self.counter = Counter()
        for fname in self.files:
            with open(fname) as f:
                try:
                    data: dict[str, dict] = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f'Failed to load {fname}')
                    continue
            for _, item in data.items():
                assert item['rating'] == item['score']
                rating = item['rating']
                quote = item['quote']
                
                if self.counter[rating] > 200: # TEMP
                    continue
                self.counter[rating] += 1
                
                review = Review(quote, rating)
                if review.isacii:
                    self.data.append(review)
        logger.info(f"{self.counter}")
        logger.info(f'Loaded {len(self.data)} reviews')

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        review = self.data[idx]
        quote, score = review.quote, review.score
        quote_ = np.frombuffer(quote.encode('ascii', 'ignore'), dtype=np.uint8)
        quote_ = torch.from_numpy(quote_.copy())
        score_ = torch.tensor(score / 5)
        # score_ = torch.rand(())
        return quote_, score_

class Net(nn.Module):
    def __init__(self, vocab_size=256, num_hidden=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        self.pre_encode = nn.Linear(vocab_size, num_hidden)
        self.lstm = nn.RNN(num_hidden, num_hidden, num_layers)
        self.linear = nn.Linear(num_hidden, 1)
    
    def forward(self, inputs, state):
        X = F.one_hot(inputs.T.long(), self.vocab_size)
        X = X.float()
        X = self.pre_encode(X)
        Y, state = self.lstm(X, state)
        output = self.linear(Y)
        output = torch.sigmoid(output)
        return output, state
    
    def begin_state(self, device, batch_size=1):
        return torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        # return (
        #     torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device),
        #     torch.zeros(self.num_layers, batch_size, self.num_hidden, device=device)
        # )

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
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    val_loss = evaluate(net, val_loader, loss, device)
    logger.info(f'Initial val loss: {val_loss:.6f}')
    
    for epoch in tqdm(range(num_epochs)):
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
        if epoch % 10 == 0:
            torch.save(net.state_dict(), f'./ckpts/model_{epoch}.pt')

@logger.catch(reraise=True)
def main():
    dataset_path = "./theater"
    dataset = TomatoDataset(dataset_path)
    if len(dataset) < 100:
        logger.warning(f"Too few data: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader, test_loader = DataLoader(train_data, batch_size=1), DataLoader(test_data, batch_size=1)

    logger.info(f'Train size: {len(train_loader)}')
    logger.info(f'Test size: {len(test_loader)}')

    # logger.debug(next(iter(train_loader))[0])

    vocab_size, num_hidden, num_layers = 256, 30, 1
    device = 'cuda:0'

    net = Net(vocab_size, num_hidden, num_layers)

    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.LSTM):
            for param in m.parameters():
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)

    net.to(device)
    
    train(net, train_loader, test_loader, 0.03, 100, device)


if __name__ == "__main__":
    main()