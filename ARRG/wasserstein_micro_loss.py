import torch
import ot
import numpy as np
from compressor import Compressor

class WassersteinLossMicro(torch.nn.Module):
    def __init__(self):
        super(WassersteinLossMicro, self).__init__()

        # Define the compressor's hyperparameters
        input_dim = 75
        output_dim = 75
        latent_dim = 2
        conditional = True
        num_labels = 2

        # Load the compression encoder
        self.compressor = Compressor(input_dim = input_dim, output_dim = output_dim, latent_dim = latent_dim, conditional = conditional, num_labels = num_labels)
        # Load the state dict
        self.compressor.load_state_dict(torch.load('compressor.pt'))
        self.compressor.eval()

    def forward(self, x, y, x_weights = None, y_weights = None):
        """
        Compute the N-dimensional Wasserstein distance between two input tensors x and y
        with x weighted by x_weights
        """
        # Extract the relevant input into the comprresor (pz)
        x = x[:,:,3]
        y = y[:,:,3]
        # Sort from least to greatest
        x, _ = torch.sort(x, dim=1)
        y, _ = torch.sort(y, dim=1)
        # Reverse the order of the exp_pz
        x = torch.flip(x, [1])
        y = torch.flip(y, [1])
        # Generate labels
        x_labels = np.full((x.shape[0], 2), [0., 1.])
        y_labels = np.full((y.shape[0], 2), [0., 1.])
        x_labels = torch.Tensor(x_labels)
        y_labels = torch.Tensor(y_labels)
        # Compute the latent dimension of the input tensors
        x_latent = self.compressor.encode(x.unsqueeze(1), x_labels)
        y_latent = self.compressor.encode(y.unsqueeze(1), y_labels)
        # Compute the cost matrix
        M = ot.dist(x_latent, y_latent, metric = 'sqeuclidean')
        # Normalize the weights to sum to one
        x_weights = x_weights / torch.sum(x_weights)
        y_weights = torch.ones(y.shape[0]) / y.shape[0]
        # Compute the Wasserstein distance
        return ot.emd2(a = x_weights, b = y_weights, M = M)
        
        """
        #from geomloss import SamplesLoss
        # Initialize sample loss object
        loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.8)
        # Normalize the weights to sum to one
        x_weights = x_weights.expand((x.shape[0], x.shape[1]))
        x_weights = x_weights / torch.sum(x_weights)
        if y_weights is None:
            y_weights = torch.ones(x_weights.shape)
            y_weights = y_weights.expand((x.shape[0], x.shape[1]))
            y_weights = y_weights / torch.sum(y_weights)
        # Compute the Wasserstein distance
        print('x_weights_shape',x_weights.shape)
        print('x shape', x.shape)
        print('y_weights_shape',y_weights.shape)
        print('y shape', y.shape)
        wass = loss(x_weights, x, y_weights, y)
        return wass.item()
        """