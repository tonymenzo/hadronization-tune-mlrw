import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import ot
import numpy as np
import tqdm
from tqdm import tqdm

import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'ARRG')))
from ARRG_macro import *

class WassersteinLoss(torch.nn.Module):
    def __init__(self, print_details=False):
        super(WassersteinLoss, self).__init__()
        self.print_details = print_details

    def forward(self, x, y, x_weights=None, y_weights=None, deltas=None, x_cdf_indices=None, y_cdf_indices=None, pre_sorted=True):
        # pre_sorted: x and y tensors are already sorted.
        # Ensure the tensors are on the same device as the weights
        device = x.device if x_weights is not None else y.device
        # Detach tensors to prevent gradient calculations
        x = x.detach().to(device)
        y = y.detach().to(device)

        if not pre_sorted:
            # Sort and compute intermediate tensors on the same device
            x_sorter = torch.argsort(x)
            y_sorter = torch.argsort(y)
            x = x[x_sorter].to(device)
            y = y[y_sorter].to(device)
            if x_weights is not None:
                x_weights = x_weights[x_sorter].to(device)
            if y_weights is not None:
                y_weights = y_weights[y_sorter].to(device)

        all_values = torch.cat((x, y))
        all_values, _ = torch.sort(all_values)

            # Compute the differences between pairs of successive values
        if deltas is None: deltas = torch.diff(all_values)

        # Get the respective positions of the values of x and y among the values of both distributions
        if x_cdf_indices is None: x_cdf_indices = torch.searchsorted(x, all_values[:-1], right=True)
        if y_cdf_indices is None: y_cdf_indices = torch.searchsorted(y, all_values[:-1], right=True)

        # Calculate the CDFs of x and y using their weights, if specified
        if x_weights is None:
            x_cdf = x_cdf_indices.float() / x.size(0)
        else:
            x_sorted_cumweights = torch.cat((torch.zeros(1, device=device), 
                                             torch.cumsum(x_weights, dim=0)))
            x_cdf = x_sorted_cumweights[x_cdf_indices] / x_sorted_cumweights[-1]

        if y_weights is None:
            y_cdf = y_cdf_indices.float() / y.size(0)
        else:
            y_sorted_cumweights = torch.cat((torch.zeros(1, device=device), 
                                             torch.cumsum(y_weights, dim=0)))
            y_cdf = y_sorted_cumweights[y_cdf_indices] / y_sorted_cumweights[-1]

        # Compute the final sum on the appropriate device
        return torch.sum(torch.abs(x_cdf - y_cdf) * deltas)

class Wasserstein_Tuner:
    def __init__(self, weight_nexus, loss = ot.wasserstein_1d, learning_rate=0.001):
        self.weight_nexus = weight_nexus
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.SGD(self.weight_nexus.parameters(), lr=learning_rate)
        self.loss = loss
        self.param_history = []

        self.param_history.append({
                    'epoch': 0,
                    'params_a': self.weight_nexus.params_a.item(),
                    'params_b': self.weight_nexus.params_b.item()
                })
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, num_epochs: int = 10, verbose=False):
        print("Beginning parameter tuning...")
        print(f"Initial (a, b): ({self.weight_nexus.params_a.item():.4f}, {self.weight_nexus.params_b.item():.4f})")
        for epoch in range(num_epochs):
            # Training loop
            train_loss = 0.0
            self.weight_nexus.train()
            self.optimizer.zero_grad()  # Zero the gradients before starting the epoch

            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
                
                self.optimizer.zero_grad()
                
                # Unpack the batch
                exp_data = batch['exp_data']
                sim_data = batch['sim_data']
                sim_accept_reject_data = batch['sim_accept_reject']
                sim_fPrel_data = batch['sim_fPrel']

                # Calculate weights
                weights = self.weight_nexus(sim_accept_reject_data, sim_fPrel_data)
                
                # Calculate Wasserstein distance
                if self.loss == ot.wasserstein_1d:
                    loss = self.loss(exp_data, sim_data, v_weights = weights/torch.sum(weights))
                elif self.loss == ot.sliced_wasserstein_distance:
                    projections = get_random_projections(d=exp_data.shape[1], n_projections=10, type_as=exp_data.dtype)
                    projections = projections.to(exp_data.device)
                    loss = self.loss(exp_data, sim_data, a=None, b=weights/torch.sum(weights), projections=projections, p=2, seed=123)
                #self.wasserstein_loss(exp_scores_data, sim_scores_data, y_weights=weights)

                # Backpropagate to accumulate gradients
                loss.backward()
                
                self.optimizer.step()
                
                # Record current values of params_a and params_b
                self.param_history.append({
                    'epoch': epoch + 1,
                    'params_a': self.weight_nexus.params_a.item(),
                    'params_b': self.weight_nexus.params_b.item()
                })

                train_loss += loss.item()

            # Print current values of params_a and params_b
            # Zero gradients for the next epoch
            self.optimizer.zero_grad()

            # Print training loss for this epoch
            avg_train_loss = train_loss / len(train_loader)
            if verbose: 
                print(f"Current (a, b): ({self.weight_nexus.params_a.item():.6f}, {self.weight_nexus.params_b.item():.6f})")
                print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4g}")
            
            if val_loader is not None:
                # Validation loop
                self.weight_nexus.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        # Unpack the batch
                        exp_data = batch['exp_data']
                        sim_data = batch['sim_data']
                        sim_accept_reject_data = batch['sim_accept_reject']
                        sim_fPrel_data = batch['sim_fPrel']

                        # Calculate weights
                        weights = self.weight_nexus(sim_accept_reject_data, sim_fPrel_data)

                        # Calculate Wasserstein distance
                        if self.loss == ot.wasserstein_1d:
                            loss = self.loss(exp_data, sim_data, v_weights = weights/torch.sum(weights))
                        elif self.loss == ot.sliced_wasserstein_distance:
                            loss = self.loss(exp_data, sim_data, a=None, b=weights/torch.sum(weights), n_projections=50, p=2, seed=123, log=False)
                        #self.wasserstein_loss(exp_scores_data, sim_scores_data, y_weights=weights)

                        val_loss += loss.item()

                # Print validation loss for this epoch
                avg_val_loss = val_loss / len(val_loader)
                if verbose: print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4g}")

                return self.weight_nexus

    def Wasserstein_flow(self, flow_loader: DataLoader, a_b_init_grid):
        """
        Generate gradient flow data using the same loss and dataloader as in training.
        """
        # Initialize gradient tensor
        a_b_gradient = torch.zeros(len(a_b_init_grid), 2)
        loss_over_batches = 0
        loss_grid = torch.zeros(len(a_b_init_grid))
        init_counter = 0

        for a_b_init in tqdm(a_b_init_grid, ncols=100, desc="Wasserstein Flow"):
            # Initialize new weight module with different initial parameters
            self.weight_nexus = LundWeight(self.weight_nexus.params_base, a_b_init, over_sample_factor=self.weight_nexus.over_sample_factor)
            self.weight_nexus.eval()  # Set the model to evaluation mode

            a_b_gradient_i = torch.zeros(2)
            for batch in flow_loader:
                # Unpack the batch
                exp_data = batch['exp_data']
                sim_data = batch['sim_data']
                sim_accept_reject_data = batch['sim_accept_reject']
                sim_fPrel_data = batch['sim_fPrel']

                # Calculate weights
                weights = self.weight_nexus(sim_accept_reject_data, sim_fPrel_data)
                
                # Calculate Wasserstein distance
                if self.loss == ot.wasserstein_1d:
                    loss = self.loss(exp_data, sim_data, v_weights = weights/torch.sum(weights))
                elif self.loss == ot.sliced_wasserstein_distance:
                    projections = get_random_projections(d=exp_data.shape[1], n_projections=10, type_as=exp_data.dtype)
                    projections = projections.to(exp_data.device)
                    loss = self.loss(exp_data, sim_data, a=None, b=weights/torch.sum(weights), projections=projections, p=2, seed=123)

                # Backpropagate to accumulate gradients
                loss.backward()
                loss_over_batches += loss.item()

                a_b_gradient_i[0] -= self.weight_nexus.params_a.grad.clone().detach()
                a_b_gradient_i[1] -= self.weight_nexus.params_b.grad.clone().detach()

            # Average gradients across batches
            a_b_gradient_i /= len(flow_loader)
            loss_over_batches /= len(flow_loader)
            # Write to the master gradient tensor
            a_b_gradient[init_counter] = a_b_gradient_i.clone()
            loss_grid[init_counter] = loss.clone().detach()
            init_counter += 1

        # Convert the gradient tensor to a numpy array
        a_b_gradient = a_b_gradient.numpy()
        loss_grid = loss_grid.numpy()
        return a_b_gradient, loss_grid
    
    def set_learning_rate(self, new_lr: float):
        self.learning_rate = new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def get_current_parameters(self):
        return {
            'params_a': self.weight_nexus.params_a.item(),
            'params_b': self.weight_nexus.params_b.item()
        }

def a_b_grid(x_range, y_range, n_points):
    """
    Creates a grid of values within a two-dimensional range and returns it in a flattened tensor.
    """
    # Create linearly spaced points for each range
    x_points = torch.linspace(x_range[0], x_range[1], n_points)
    y_points = torch.linspace(y_range[0], y_range[1], n_points)

    # Create a meshgrid from the x and y points
    x_grid, y_grid = torch.meshgrid(x_points, y_points, indexing='ij')

    # Flatten the grid and stack the coordinates
    grid_flattened = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)

    return grid_flattened

def get_random_projections(d, n_projections, seed=123, type_as=None):
    r"""
    Generates n_projections samples from the uniform on the unit sphere of dimension :math:`d-1`: :math:`\mathcal{U}(\mathcal{S}^{d-1})`
    Adapted from ot library
    Parameters
    ----------
    d : int
        dimension of the space
    n_projections : int
        number of samples requested
    seed: int or RandomState, optional
        Seed used for numpy random number generator
    backend:
        Backend to use for random generation

    Returns
    -------
    out: ndarray, shape (d, n_projections)
        The uniform unit vectors on the sphere

    Examples
    --------
    >>> n_projections = 100
    >>> d = 5
    >>> projs = get_random_projections(d, n_projections)
    >>> np.allclose(np.sum(np.square(projs), 0), 1.)  # doctest: +NORMALIZE_WHITESPACE
    True

    """
    
    # Set the seed for torch if provided
    if seed is not None:
        torch.manual_seed(seed)

    # Generate random projections using torch
    projections = torch.randn(d, n_projections, dtype=type_as) if type_as is not None else torch.randn(d, n_projections)
    
    # Normalize projections along the second dimension
    projections = projections / torch.sqrt(torch.sum(projections**2, dim=0, keepdim=True))
    
    return projections

