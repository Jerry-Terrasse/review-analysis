import torch
import json
from matplotlib import pyplot as plt

class Glove:
    def __init__(self, fname: str, dim: int, cache: bool = False, reserved: list[str] = []) -> None:
        self.dim = dim
        self.word2idx: dict[str, int] = {'<ukn>': 0}
        self.idx2word: list[str] = ['<ukn>']
        self.idx2vec_: list[list[float]] = [[0] * dim] # list vec for json dump
        self.idx2vec: list[torch.Tensor] = [torch.zeros(dim)]
        if cache:
            self.load(fname)
            return
        for word in reserved:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
            self.idx2vec.append(torch.zeros(dim))
            self.idx2vec_.append([0] * dim)
        with open(fname, 'r') as f:
            for line in f:
                word, *vector = line.split()
                assert len(vector) == dim
                vector_ = list(map(float, vector))
                vector = torch.tensor(vector_)
                self.word2idx[word] = len(self.idx2word)
                self.idx2word.append(word)
                self.idx2vec.append(vector)
                self.idx2vec_.append(vector_)
    def dump(self, fname: str) -> None:
        json.dump({
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'idx2vec': self.idx2vec_,
        }, open(fname, 'w'), indent=None)
    def load(self, fname: str) -> None:
        data = json.load(open(fname, 'r'))
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx2vec_ = data['idx2vec']
        self.idx2vec = list(map(torch.tensor, self.idx2vec_))
        assert len(self.idx2word) == len(self.idx2vec)
        assert len(self.idx2word) == len(self.word2idx)
        assert len(self.idx2vec[0]) == self.dim

def plot_loss(loss: list[float], fname: str) -> None:
    plt.figure()
    plt.plot(loss,'b', label = 'loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def plot_multi_loss(loss: list[list[float]], labels: list[str], fname: str) -> None:
    plt.figure()
    for i in range(len(loss)):
        plt.plot(loss[i], label = labels[i])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(fname)
    plt.close()

def plot_hist(data: list[int], bins: list, fname: str) -> None:
    plt.figure()
    plt.hist(data, bins=bins)
    plt.savefig(fname)
    plt.close()