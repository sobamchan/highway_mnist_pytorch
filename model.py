import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Linear(nn.Module):

    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)


class Bottle(nn.Module):

    def forward(self, input_):
        input_size_l = len(input_.size())
        if input_size_l <= 2:
            return super(Bottle, self).forward(input_)

        size = input_.size()[:2]
        out_ = super(Bottle, self).forward(input_.view(size[0] * size[1], -1))
        return out_.view(size[0], size[1], -1)


class BottleLinear(Bottle, Linear):
    pass


class Highway(nn.Module):

    def __init__(self, d, activation_func=nn.ReLU(), bias=True):
        super(Highway, self).__init__()
        self.linear_h = BottleLinear(d, d, bias)
        self.linear_t = BottleLinear(d, d, bias)
        self.f = activation_func

    def forward(self, x):
        f_val = self.f(self.linear_h(x))
        sig_val = F.sigmoid(self.linear_t(x))
        gate_val = 1 - sig_val
        return f_val * sig_val + x * gate_val


class HighwayNet(nn.Module):

    def __init__(self):
        super(HighwayNet, self).__init__()
        self.hi1 = Highway(784)
        self.hi2 = Highway(784)
        self.fc1 = BottleLinear(784, 50)
        self.fc2 = BottleLinear(50, 10)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.hi1(x)
        x = self.hi2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x).view(batch_size, -1)
        return F.log_softmax(x, dim=1)
