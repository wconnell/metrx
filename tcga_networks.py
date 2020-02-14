import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class EmbeddingNet(nn.Module):
    def __init__(self, n_features):
        super(EmbeddingNet, self).__init__()
        self.n_features = n_features
        self.fc = nn.Sequential(OrderedDict([
                                ('linear1', nn.Linear(self.n_features, 2000)),
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

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)
    
    def get_loss(self, x1, x2, target, loss_fn):
        outputs = self.forward(x1, x2)
        loss_inputs = outputs
        
        if target is not None:
            target = (target,)
            loss_inputs += target
            
        loss_outputs = loss_fn(*loss_inputs)
        loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
        return loss.view(-1)
    
    def get_dist(self, x1, x2, dist_fn):
        outputs = self.forward(x1, x2)
        return dist_fn(*outputs)
        
