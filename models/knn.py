import torch
from torch import nn,Tensor
from torch.utils.data import Dataset
from sklearn.neighbors import KDTree,BallTree
from utils.funcs import normedChebyshev

class KNN(object):

    def __init__(self, dataset: Dataset):
        inputs      = []
        targets     = []
        for x,y in dataset:
            inputs.append(x)
            targets.append(y)
        inputs = torch.stack( inputs )
        self.tree = BallTree( inputs, metric=normedChebyshev )
        self.targets = torch.tensor(targets)

    def __call__(self, x : Tensor) -> Tensor:
        dist,index = self.tree.query( x )
        return torch.tensor( dist ).flatten(),self.targets[torch.tensor( index ).flatten()]