import torch
from torch import nn, Tensor
from scipy.spatial.distance import cosine
from typing import Optional
def mean(x : Tensor, dim : int = 1, weights: Optional[ None, Tensor ] = None) -> Tensor:
    '''
        Apply the mean to the index that represents the number of words
    '''
    return x.mean(dim=dim) if weights is not None else x.mean(dim=dim) * weights

def centroid(x : Tensor, index: int = 1) -> Tensor: 
    '''
        Calculate the mean vector and return the one closest to the center
    '''
    from utils.funcs import normedChebyshev
    mu = mean( x , index )
    best = torch.argmin([ normedChebyshev( mu, x_i ) for x_i in x[ :, index ] ])
    return x[:, [best], :]

def diff(x : Tensor) -> Tensor:
    '''
        Take the difference between the min and max of each dimension.
        This will produce only positive values.

        NOTE: This kind of seems dumb, but I'm leaving it for now.
    '''
    pass

