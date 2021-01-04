import torch
import torch.nn as nn
import torch.nn.functional as F

# for LDAM Loss
class FCNorm(nn.Module):
    def __init__(self, num_features, num_classes):
        super(FCNorm, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.linear(F.normalize(x), F.normalize(self.weight))
        return out


class FC2(nn.Module):
    def __init__(self, num_features, num_classes, bias):
        super(FC2, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.bias_flag = bias

    def forward(self, x):
        x = nn.Linear(self.num_features, 2048, bias=self.bias_flag)(x)
        out = nn.Linear(2048, self.num_classes, bias=self.bias_flag)(x)
        return out