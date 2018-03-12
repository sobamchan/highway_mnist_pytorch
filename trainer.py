import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import datasets, transforms


def get_data_loader(args):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.1307, ),
                                                  (0.3081, ))])
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.use_cuda else {}
    train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/',
                                                              train=True,
                                                              download=True,
                                                              transform=tf),
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               **kwargs)

    test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data/',
                                                             train=False,
                                                             download=True,
                                                             transform=tf),
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              **kwargs)

    return train_loader, test_loader


class Trainer(object):

    def __init__(self, args):
        self.args = args
        train_loader, test_loader = get_data_loader(args)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def set_model(self, model):
        args = self.args
        if args.use_cuda:
            self.model = model.cuda()
        else:
            self.model = model

    def set_optimizer(self, optimizer):
        args = self.args
        self.optimizer = optimizer(self.model.parameters(), lr=args.lr)

    def train(self):
        args = self.args
        losses = []
        for i_epoch in range(1, args.epoch + 1):
            loss = self.train_one_epoch(i_epoch)
            losses.append(loss)
            test_loss, test_acc = self.test()
            if i_epoch % 10:
                print('{}th epoch, train loss: {}'.format(i_epoch,
                                                          np.mean(losses)))
                print('{}th epoch, test loss: {}'.format(i_epoch, test_loss))
                print('{}th epoch, test acc: {}'.format(i_epoch, test_acc))

    def train_one_epoch(self, i_epoch):
        args = self.args
        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            size = data.size()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            data = data.view(size[0], -1, size[2] * size[3])
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
        return loss.data.tolist()[0]

    def test(self):
        args = self.args
        self.model.eval()
        losses = []
        correct = 0
        for data, target in self.test_loader:
            size = data.size()
            if args.use_cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data, volatile=True), Variable(target)
            data = data.view(size[0], -1, size[2] * size[3])
            output = self.model(data)
            loss = F.nll_loss(output, target, size_average=False)
            losses.append(loss.data.tolist()[0])
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        accuracy = 100. * correct / len(self.test_loader.dataset)
        return np.mean(losses), accuracy
