
import copy
import random
from functools import wraps

import torch
from torch import nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchvision import transforms as T

# helper functions

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

def default(val, def_val):
    return def_val if val is None else val

def get_module_device(module):
    return next(module.parameters()).device


# MLP class for projector and predictor

class MLP(nn.Module):
    def __init__(self, dim, projection_size, hidden_size = 512):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(dim, hidden_size),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, projection_size)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


    def forward(self, x):

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.net(x)

        return x


class BYOL(nn.Module):
    """
    Build a BYOL model. https://arxiv.org/abs/2006.07733
    """
    def __init__(self,dim=512, pred_dim=128, m=0.996):
        """
        encoder_q: online network
        encoder_k: target network
        dim: feature dimension (default: 4096)
        pred_dim: hidden dimension of the predictor (default: 256)
        """
        super(BYOL, self).__init__()
        self.unet = smp.Unet(classes=2)
        self.encoder_q = self.unet.encoder

        self.encoder_k = smp.Unet(classes=2).encoder

        self.recoder =  smp.Unet(classes=3)
        self.recoder.encoder = self.encoder_q

        self.m = m

        # projector
        # encoder_dim = self.encoder_q.fc.weight.shape[1]
        self.encoder_q_mlp = MLP(dim,projection_size = pred_dim)
        self.encoder_k_mlp =  MLP(dim,projection_size = pred_dim)

        self.predictor = nn.Sequential(nn.Linear(pred_dim, dim),
                                       nn.BatchNorm1d(dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(dim, pred_dim))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.encoder_q_mlp.parameters(), self.encoder_k_mlp.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def forward(self, x1, x2,x3):
        """
        Input:
            x1: first views of images
            x2: second views of images
        """

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

        # print(self.encoder_q(x1)[-1].shape)
        # exit()
        p1 = self.predictor(self.encoder_q_mlp(self.encoder_q(x1)[-1]))
        z2 = self.encoder_k_mlp(self.encoder_k(x2)[-1])

        p2 = self.predictor( self.encoder_q_mlp(self.encoder_q(x2)[-1]))  # NxC
        z1 = self.encoder_k_mlp(self.encoder_k(x1)[-1])

        result1 = self.unet(x1)
        result2 = self.unet(x2)
        result3 = self.recoder(x3)

        return p1, p2, z1.detach(), z2.detach(),result1,result2,result3



if __name__ == '__main__':

    model = BYOL()

    images = [torch.rand(4,3,224,224),
              torch.rand(4,3,224,224)]

    p1, p2, z1, z2,_,_ = model(x1=images[0], x2=images[1])
    criterion = nn.CosineSimilarity()
    loss = 2 - (criterion(p1, z2).mean() + criterion(p2, z1).mean())
    print(loss)
    exit()