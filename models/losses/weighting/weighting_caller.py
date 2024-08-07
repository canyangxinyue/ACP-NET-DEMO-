import torch
import torch.nn as nn


class WeightingCaller(nn.Module):
    
    def __init__(self, backward_method, losses, **kwargs):
        super(WeightingCaller, self).__init__()
        self.backward_method = backward_method
        self.losses = losses
        self.weighting_config=kwargs['weighting_config']
        self.weight=0
        
    def item(self):
        return (self.losses.detach().cpu().numpy()*self.weight).sum()
    
    def items(self):
        return self.losses
        
    def backward(self):
        self.weight=self.backward_method(self.losses, **self.weighting_config)
