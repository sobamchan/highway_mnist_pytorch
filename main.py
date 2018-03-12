import argparse
import torch
import torch.optim as optim
from trainer import Trainer
from model import HighwayNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=bool, default=True)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.use_cuda and torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    trainer = Trainer(args)

    model = HighwayNet()
    trainer.set_model(model)
    optimizer = optim.SGD
    trainer.set_optimizer(optimizer)

    trainer.train()


if __name__ == '__main__':
    main()
