import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
#         self.fc = nn.Sequential(nn.Linear(60483, 2000),
#                                  nn.PReLU(),
#                                  nn.Linear(2000, 500),
#                                  nn.PReLU(),
#                                  nn.Linear(500, 2)
#                                )
        self.fc = nn.Sequential(OrderedDict([
                                ('linear1', nn.Linear(60483, 2000)),
                                ('relu1', nn.PReLU()),
                                ('linear2', nn.Linear(2000, 500)),
                                ('relu2', nn.PReLU()),
                                ('linear3', nn.Linear(500, 250)),
                                ('relu3', nn.PReLU()),
                                ('linear4', nn.Linear(250, 100)),
                                ('relu4', nn.PReLU()),
                                ('linear5', nn.Linear(100, 50)),
                                ('relu5', nn.PReLU()),
                                ('linear6', nn.Linear(50, 10)),
                                ('relu6', nn.PReLU()),
                                ('linear7', nn.Linear(10, 2))
                                ]))

    def forward(self, x):
        output = self.fc(x)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, interp):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        if interp:
            return torch.stack((output1, output2))
        else:
            return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
    
