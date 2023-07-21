import torch
import sys

from train import *
from preprocess import clean

@logger.catch
def test(model_path: str):
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
    net.load_state_dict(torch.load(model_path))
    net.to(device)
    
    while True:
        text = input('Input: ')
        if text == 'exit':
            break
        text = clean(text)
        if text == '':
            logger.warning('Input Invalid')
            continue
        text = [dataset.glove.word2idx.get(word, 0) for word in text]
        text = torch.tensor(text, dtype=torch.long).unsqueeze(0).to(device)
        state = net.begin_state(device)
        output, state = net(text, state)
        logger.info(f'Output: {output[-1].item()}')

if __name__ == '__main__':
    model_path = sys.argv[1]
    test(model_path)