import torch
from torch import nn,Tensor

# Create Model
class Simple(nn.Module):
    
    def __init__(self,INP_SIZE: int,N_CLASSES: int):
        super(TweetClf,self).__init__()
        self.f = nn.Sequential(
            nn.ReLU(),
            nn.Linear(INP_SIZE,INP_SIZE),
            nn.Dropout(),
            nn.GELU(),
            nn.Linear(INP_SIZE,N_CLASSES),
        )
        
    def forward(self,x:Tensor) -> Tensor:return self.f(x)