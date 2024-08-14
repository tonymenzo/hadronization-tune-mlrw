import torch
import matplotlib.pyplot as plt
import ot
import numpy as np

class WassersteinLoss(torch.nn.Module):
    def __init__(self, print_details=False):
        super(WassersteinLoss, self).__init__()
        self.print_details = print_details

    def forward(self, x, y, x_weights=None, y_weights=None):
        x.detach()
        y.detach()
        x_sorter = torch.argsort(x)
        y_sorter = torch.argsort(y)
        
        all_values = torch.cat((x, y))
        all_values, _ = torch.sort(all_values)
        
        # Compute the differences between pairs of successive values
        deltas = torch.diff(all_values)
        
        # Get the respective positions of the values of x and y among the values of both distributions
        x_cdf_indices = torch.searchsorted(x[x_sorter].detach(), all_values[:-1], right=True)
        y_cdf_indices = torch.searchsorted(y[y_sorter].detach(), all_values[:-1], right=True)

        # Calculate the CDFs of x and y using their weights, if specified
        if x_weights is None:
            x_cdf = x_cdf_indices.float() / x.size(0)
        else:
            x_sorted_cumweights = torch.cat((torch.zeros(1, device=x.device), 
                                             torch.cumsum(x_weights[x_sorter], dim=0)))
            x_cdf = x_sorted_cumweights[x_cdf_indices] / x_sorted_cumweights[-1]
        
        if y_weights is None:
            y_cdf = y_cdf_indices.float() / y.size(0)
        else:
            cumsum = torch.cumsum(y_weights[y_sorter], dim=0)
            y_sorted_cumweights = torch.cat((torch.zeros(1, device=y.device), 
                                         cumsum))
            y_cdf = y_sorted_cumweights[y_cdf_indices] / y_sorted_cumweights[-1]
        return torch.sum(torch.abs(x_cdf - y_cdf) * deltas)
    
class WassersteinLoss_POT(torch.nn.Module):
    def __init__(self):
        super(WassersteinLoss_POT, self).__init__()

    def forward(self, x, y, x_weights=None, y_weights=None):
        # Compute Wasserstein distance
        if x_weights==None:
            x_weights = torch.ones(len(x))
        x_weights=x_weights/torch.sum(x_weights) # weights have to be normalized
        if y_weights==None:
            y_weights = torch.ones(len(y))
        norm_y_weights=y_weights/torch.sum(y_weights) # weights have to be normalized
        loss = ot.wasserstein_1d(x, y, u_weights=x_weights, v_weights=norm_y_weights)
        return loss