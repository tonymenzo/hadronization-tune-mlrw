import torch
import ot

class WassersteinLoss(torch.nn.Module):
    def __init__(self, p):
        super(WassersteinLoss, self).__init__()
        self.p = p

    def forward(self, x, y, x_weights =None, y_weights = None):
        """
        Compute the one-dimensional Wasserstein distance between two input tensors x and y
        with x weighted by x_weights
        """
        return ot.wasserstein_1d(x, y, u_weights = x_weights, v_weights = y_weights, p = self.p)