import torch
from torch import Tensor
from scipy.spatial.distance import *
import numpy as np
def normedChebyshev(x:Tensor,y:Tensor) -> Tensor:
    '''
        Calculate the Chebyshev distance and divide by the norm.
        This will compress the output into a set [0,1]
        Think of this simlar to the cosine similarity measure - except more appropriate for higher dimensional spaces.


        REF: https://www.researchgate.net/publication/30013021_On_the_Surprising_Behavior_of_Distance_Metric_in_High-Dimensional_Space
    '''
    assert x.size == y.size, 'Tensors must be of same shape when calculating the Normed Chebyshev Distance!'
    return chebyshev(x,y)/(np.linalg.norm(x)*np.linalg.norm(x))