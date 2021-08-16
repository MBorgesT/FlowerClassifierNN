import torch
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import datasets
from torchvision import transforms as T


DATA_DIR = './data/'


def get_data(data_dir, transform=None):
    data = datasets.ImageFolder(data_dir, transform=transform)
    return shuffle(data)


def get_k_fold_data(k, i, dataset):
    assert k > 1

    k_size = int(len(dataset) / k)
    train_ids = list(range(k_size * i)) + list(range(k_size * (i + 1), len(dataset)))
    valid_ids = range(k_size * i, k_size * (i + 1))

    train_set = Subset(dataset, train_ids)
    valid_set = Subset(dataset, valid_ids)

    return train_set, valid_set


def get_net():
    input_size = 320 * 240 * 3
    net = nn.Sequential(
        nn.Linear(input_size, 2048),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(.1),
        nn.Linear(256, 5),
        nn.LogSoftmax(dim=1)
    )

    if torch.cuda.is_available():
        net.cuda()

    return net


def train(net, train_iter, valid_iter, loss, num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, valid_ls = [], []
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        train_l_epoch, valid_l_epoch = [], []

        # train
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
            train_l_epoch.append(l.item())
        train_ls.append(sum(train_l_epoch) / len(train_l_epoch))
        
        # validation
        with torch.no_grad():
            for X, y in valid_iter:
                valid_l_epoch.append(loss(net(X), y).item())
        valid_ls.append(sum(valid_l_epoch) / len(valid_l_epoch))

        print(f'e: {epoch}\ttrain loss: {train_ls[-1]:.4f}\tvalid loss: {valid_ls[-1]:.4f}')
    
    return train_ls, valid_ls


def k_fold(k, dataset, num_epochs, learning_rate, weight_decay, batch_size):
    train_l, valid_l = [], []
    net = get_net()
    loss = nn.CrossEntropyLoss()
    for i in range(k):
        print('k:', i)
        train_set, valid_set = get_k_fold_data(k, i, dataset)
        train_iter = DataLoader(train_set, batch_size=batch_size)
        valid_iter = DataLoader(valid_set, batch_size=batch_size)

        k_train_ls, k_valid_ls = train(net, train_iter, valid_iter, loss, num_epochs, learning_rate, weight_decay, batch_size)

        train_l += k_train_ls
        valid_l += k_valid_ls

    plot_loss_graph(train_l, valid_l)
    return sum(train_l) / (k * num_epochs), sum(valid_l) / (k * num_epochs)


def plot_loss_graph(train_l, valid_l):
    _, ax = plt.subplots()

    ax.plot(train_l)
    ax.plot(valid_l)
    ax.set(xlabel='k', ylabel='sgd', title='Loss over k iterations')

    plt.show()


k, num_epochs, lr, weight_decay, batch_size = 5, 12, .05, 0, 64

transform = T.Compose([
    T.Resize((320, 240)),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    T.Lambda(lambda x: torch.flatten(x))
])
dataset = get_data(DATA_DIR, transform)

train_l, valid_l = k_fold(k, dataset, num_epochs, lr, weight_decay, batch_size)

print(f'{k}-fold validation: avg train CEL:{float(train_l):f},'f'avg valid CEL:{float(valid_l):f}')

