import torch
import ot

class WassersteinLossMacro(torch.nn.Module):
    def __init__(self, p):
        super(WassersteinLossMacro, self).__init__()
        self.p = p

    def forward(self, x, y, x_weights = None, y_weights = None):
        """
        Compute the one-dimensional Wasserstein distance between two input tensors x and y
        with x weighted by x_weights
        """
        # The weights must be normalized for the Wasserstein loss
        x_weights = x_weights / torch.sum(x_weights)
        y_weights = torch.ones(y.shape[0]) / y.shape[0]
        # Compute the Wasserstein distance
        return ot.wasserstein_1d(x, y, u_weights = x_weights, v_weights = y_weights, p = self.p)