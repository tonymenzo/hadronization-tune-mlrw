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

import utils
from utils import *

class Wasserstein_Tuner:
    def __init__(self, params_base, over_sample_factor, params_init=None,
                 loss=ot.wasserstein_1d, learning_rate=0.01, optimizer="Adam", scheduler=None):
        self.params_base = params_base
        if params_init is None: self.params_init = params_base
        self.params_init = params_init
        self.over_sample_factor = over_sample_factor
        self.weight_nexus = LundWeight(self.params_base, self.params_init, 
                                       over_sample_factor = self.over_sample_factor).to("cuda")
        
        self.lr = learning_rate
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.weight_nexus.parameters(), lr=learning_rate, amsgrad=True)
        else:
            self.optimizer = torch.optim.SGD(self.weight_nexus.parameters(), lr=learning_rate)
        self.loss = loss
        self.scheduler = scheduler
        self.param_history = []
        self.wass_distances = []
        self.cum_min_wass_distances = []
        self.params_best = None  # Placeholder for best parameters
        self.device="cuda"
        
        # Record initial parameter values
        self.param_history.append({
            'epoch': 0,
            'params_a': self.weight_nexus.params_a.item(),
            'params_b': self.weight_nexus.params_b.item()
        })
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None, num_epochs: int = 10, verbose=False, 
              n_projections=128, patience=100, delta=0, grad_ave=True):
        print("Beginning parameter tuning...")
        print(f"Initial (a, b): ({self.weight_nexus.params_a.item():.4f}, {self.weight_nexus.params_b.item():.4f})")
        if self.scheduler is None: 
            print("No learning rate scheduler provided, using constant learning rate:", self.lr)

        best_loss = float('inf')  # Initialize best loss
        epochs_without_improvement = 0  # Counter for early stopping
        best_epoch = 0  # To store the epoch with the best loss

        for epoch in tqdm(range(num_epochs), desc="Wasserstein Tuning", dynamic_ncols=True):
            # Training loop
            train_loss = 0.0
            self.weight_nexus.train()
            self.optimizer.zero_grad()  # Zero the gradients before starting the epoch

            #loss = torch.tensor(0.0).to("cuda")
            for batch in train_loader:
                #batch = {k: v.to(self.device).detach() for k, v in batch.items()}
                # Unpack the batch
                exp_data = batch['exp_data'].detach().to(self.device)
                sim_data = batch['sim_data'].detach().to(self.device)
                sim_accept_reject_data = batch['sim_accept_reject'].detach().to(self.device)
                sim_fPrel_data = batch['sim_fPrel'].detach().to(self.device)
                
                weights = self.weight_nexus(sim_accept_reject_data, sim_fPrel_data).to(self.device)
                if sim_accept_reject_data.ndim < 4:
                    if self.loss == ot.wasserstein_1d:
                        loss = self.loss(exp_data, sim_data, v_weights=weights/torch.sum(weights))
                    elif self.loss == ot.sliced_wasserstein_distance:
                        loss = self.loss(exp_data, sim_data, a=None, b=weights/torch.sum(weights),
                                          n_projections=n_projections, p=2)
                else:
                    weights = torch.prod(weights,dim=0)
                    #exp_data = exp_data[0]
                    #exp_data = exp_data.reshape(exp_data.shape[0]*exp_data.shape[1], exp_data.shape[2])
                    M = exp_data.shape[1]
                    exp_data = exp_data.permute(1, 0, 2).reshape(M, -1)
                    M = sim_data.shape[1]
                    sim_data = sim_data.permute(1, 0, 2).reshape(M, -1)
                    #M = sim_data.shape[1]
                    #sim_data = sim_data.permute(1, 0, 2).reshape(M, -1)#.reshape(sim_data.shape[0]*sim_data.shape[1], sim_data.shape[2])
                    loss = self.loss(exp_data, sim_data, a=None, b=weights/torch.sum(weights), n_projections=n_projections, p=2)
                loss.backward()
                # If grad_ave is False, update weights after each batch
                if not grad_ave:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # Record current values of params_a and params_b
                train_loss += loss.clone().item()
                
                del exp_data 
                del sim_data
                del sim_accept_reject_data
                del sim_fPrel_data

            # If grad_ave is True, average gradients over all batches and update weights
            if grad_ave:
                for param in self.weight_nexus.parameters():
                    if param.grad is not None:
                        param.grad /= 1#len(train_loader)
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Step the learning rate scheduler, if one is provided
            if self.scheduler:
                self.scheduler.step()

            self.param_history.append({
                    'epoch': epoch + 1,
                    'params_a': self.weight_nexus.params_a.item(),
                    'params_b': self.weight_nexus.params_b.item()
                })
            
            # Calculate average training loss for the epoch
            avg_train_loss = train_loss / len(train_loader)
            self.wass_distances.append(avg_train_loss)

            # Update cumulative minimum wasserstein loss history
            if len(self.cum_min_wass_distances) == 0:
                self.cum_min_wass_distances.append(avg_train_loss)
            else:
                self.cum_min_wass_distances.append(min(avg_train_loss, self.cum_min_wass_distances[-1]))

            # Check if current loss is the best one and save the parameters if it is
            if avg_train_loss < best_loss - delta:
                best_loss = avg_train_loss
                self.params_best = [self.weight_nexus.params_a.item(), self.weight_nexus.params_b.item()]
                epochs_without_improvement = 0  # Reset patience
                best_epoch = epoch
            else:
                epochs_without_improvement += 1  # Increment counter if no improvement

        return self.weight_nexus
        
    def Wasserstein_flow(self, flow_loader: DataLoader, a_b_init_grid, n_projections=128):
        """
        Generate gradient flow data using the same loss and dataloader as in training.
        """
        # Initialize gradient tensor
        a_b_gradient = torch.zeros(len(a_b_init_grid), 2, len(flow_loader), device=self.device)
        loss_grid = torch.zeros(len(a_b_init_grid), len(flow_loader), device=self.device)

        for j, batch in enumerate(flow_loader):
            # Move batch data to the appropriate device
            exp_data = batch['exp_data'].detach().to(self.device)
            sim_data = batch['sim_data'].detach().to(self.device)
            sim_accept_reject_data = batch['sim_accept_reject'].detach().to(self.device)
            sim_fPrel_data = batch['sim_fPrel'].detach().to(self.device)

            # Initialize tqdm for the inner loop
            for idx, a_b_init in enumerate(tqdm(a_b_init_grid, ncols=100, desc="Wasserstein Flow")):
                # Initialize new weight module with different initial parameters
                self.weight_nexus = LundWeight(self.weight_nexus.params_base, a_b_init, 
                                               over_sample_factor=self.weight_nexus.over_sample_factor).to(self.device)
                self.weight_nexus.eval()  # Set the model to evaluation mode

                # Calculate weights for this batch
                weights = self.weight_nexus(sim_accept_reject_data, sim_fPrel_data).to(self.device)
                total_weights = torch.sum(weights)

                # Calculate Wasserstein distance using total accumulated weights
                if self.loss == ot.wasserstein_1d:
                    loss = self.loss(exp_data, sim_data, v_weights = weights/total_weights, p=2)
                elif self.loss == ot.sliced_wasserstein_distance:
                    loss = self.loss(exp_data, sim_data, b = weights/total_weights, 
                                     n_projections=n_projections, p=2)
                
                loss_grid[idx, j] = loss.clone().item()
                # Backpropagate to accumulate gradients
                loss.backward()

                # Accumulate loss and gradients
                a_b_gradient[idx, 0, j] = -self.weight_nexus.params_a.grad.clone().detach()
                a_b_gradient[idx, 1, j] = -self.weight_nexus.params_b.grad.clone().detach()
                
                self.optimizer.zero_grad()

        # Convert the gradient tensor to a numpy array
        a_b_gradient = torch.mean(a_b_gradient, axis=2).cpu().numpy()
        loss_grid = torch.mean(loss_grid, axis=1).cpu().numpy()
        return a_b_gradient, loss_grid

    
    def plot_a_b_history(self, save_name):
        plt.figure() 
        
        a_history_s = [p['params_a'] for p in self.param_history]
        b_history_s = [p['params_b'] for p in self.param_history]

        plt.plot(a_history_s, b_history_s, marker='o', markersize=2)#, label="2D Sliced")
        plt.scatter(self.params_base.cpu().detach().numpy()[0], self.params_base.cpu().detach().numpy()[1], marker='o', s=100, color="red", label="Base")
        plt.scatter(0.68, 0.98, marker='x', s=100, color="green", label="Truth")
        plt.scatter(self.params_best[0], self.params_best[1], marker='*', s=100, color="blue", label="Best")

        plt.axvline(x=0.68, color='green', linestyle='--')
        plt.axhline(y=0.98, color='green', linestyle='--')

        plt.title(f"Tuning history")
        plt.xlabel("a")
        plt.ylabel("b")
        plt.legend()
        
        print("Saving example tuning history to:", save_name)
        plt.savefig(save_name)
        plt.show()         
    
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

def bootstrap_tune(exp_data, sim_data, batch_size, sim_fPrel, sim_accept_reject, N_events, num_epochs, k, tuner,
                   tuned_params=None, tuned_best_params=None, n_projections=128, patience=100, delta=0, optimizer="Adam",
                   device="cuda", saveName="bootstrap_params.npy"):
    if tuned_params is None:
        tuned_params = []  # To store (a, b) parameters for each bootstrapped dataset
    if tuned_best_params is None:
        tuned_best_params = []
    tuned_params = list(tuned_params)
    tuned_best_params = list(tuned_best_params)
 
    params_base = tuner.weight_nexus.params_base
    params_init = tuner.params_init
    over_sample_factor = tuner.weight_nexus.over_sample_factor
    learning_rate = tuner.lr
    loss = tuner.loss
    # Loop over k bootstrapped datasets
    for i in range(k):
        # Sample indices with replacement
        boot_indices = np.random.choice(np.arange(N_events), size=N_events, replace=True)
        
        # Create bootstrapped datasets
        exp_data_boot = exp_data[boot_indices].to(device)
        sim_data_boot = sim_data[boot_indices].to(device)
        sim_fPrel_boot = sim_fPrel[boot_indices].to(device)
        sim_accept_reject_boot = sim_accept_reject[boot_indices].to(device)
        
        # Create a bootstrapped dataset and DataLoader
        train_dataset = CombinedDataset(exp_data_boot, sim_data_boot, sim_fPrel_boot, sim_accept_reject_boot, pre_sorted=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Train the tuner
        tuner.train(train_loader, val_loader=None, num_epochs=num_epochs, n_projections=n_projections,
                    patience=patience, delta=delta, grad_ave=True)
        
        # Get the tuned (a, b) parameters and store them
        tuned_a_b = np.array([tuner.weight_nexus.params_a.cpu().detach().numpy(), tuner.weight_nexus.params_b.cpu().detach().numpy()])
        tuned_params.append(tuned_a_b)
        tuned_best_a_b = np.array([tuner.params_best[0], tuner.params_best[1]])
        tuned_best_params.append(tuned_best_a_b) 
        
        print(f"Bootstrap iteration {i+1}/{k}: Tuned (a, b) = {tuned_a_b}")
        np.save(saveName, np.array(tuned_params))
        np.save(saveName[:-4]+"_best.npy", np.array(tuned_best_params))
        
        if i ==0: tuner.plot_a_b_history(saveName[:-4]+"_tuning_history_debug.png")
        
        # reset tuner
        del tuner
        tuner = Wasserstein_Tuner(params_base, over_sample_factor, params_init=params_init, learning_rate=learning_rate, 
                                  optimizer=optimizer, loss=loss)
    return np.array(tuned_params), np.array(tuned_best_params)

from matplotlib.patches import Ellipse

def plot_confidence_ellipse(tuned_params, ax=None, n_std=1.96, facecolor='none', save_path=None, **kwargs):
    """
    Create a plot of a confidence ellipse for the bootstrapped parameters.
    
    Args:
        tuned_params: k x 2 array where each row contains a pair of [a, b].
        ax: Matplotlib Axes object where the ellipse will be drawn.
        n_std: Number of standard deviations to determine the ellipse size.
        facecolor: Color of the ellipse (default is transparent).
        **kwargs: Other keyword arguments for the ellipse.
    """
    # Mean of the bootstrapped parameters (center of the ellipse)
    means = np.mean(tuned_params, axis=0)
    print("Number of data points:", len(tuned_params))
    print("Means:", means)
    
    # Covariance matrix
    cov = np.cov(tuned_params, rowvar=False)
    
    # Eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Calculate angle of rotation for the ellipse
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    
    # Width and height of the ellipse
    width, height = 2 * n_std * np.sqrt(eigvals)
    
    # Create the ellipse
    ellipse = Ellipse(xy=means, width=width, height=height, angle=angle, facecolor=facecolor, **kwargs)

    if ax is None:
        fig, ax = plt.subplots()
        ax.add_patch(ellipse)
        ax.set_xlim(means[0] - 3 * width, means[0] + 3 * width)
        ax.set_ylim(means[1] - 3 * height, means[1] + 3 * height)
        ax.set_xlabel('Parameter a')
        ax.set_ylabel('Parameter b')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()
    # Add the ellipse to existing plot
    else: ax.add_patch(ellipse)

class Wasserstein_Minibatch_Tuner:
    def __init__(self, params_base, over_sample_factor, params_init=None,
                 learning_rate=0.01, optimizer="Adam", scheduler=None, loss=ot.wasserstein_1d):
        self.params_base = params_base
        self.params_init = params_base if params_init is None else params_init
        self.over_sample_factor = over_sample_factor
        self.weight_nexus = LundWeight(self.params_base, self.params_init, 
                                       over_sample_factor=self.over_sample_factor).to("cuda")
        
        self.lr = learning_rate
        self.optimizer = torch.optim.Adam(self.weight_nexus.parameters(), lr=learning_rate, amsgrad=True) if optimizer == "Adam" else torch.optim.SGD(self.weight_nexus.parameters(), lr=learning_rate)
        self.scheduler = scheduler
        self.param_history = []
        self.wass_distances = []
        self.cum_min_wass_distances = []
        self.params_best = None  # Placeholder for best parameters
        self.device = "cuda"
        self.loss = loss
        
        # Record initial parameter values
        self.param_history.append({
            'epoch': 0,
            'params_a': self.weight_nexus.params_a.item(),
            'params_b': self.weight_nexus.params_b.item()
        })
    
    def minibatch_wasserstein_distance(self, exp_data, sim_data, weights, n_projections=128):
        """ Compute the minibatch Wasserstein distance using the incomplete estimator. """
        # Normalize weights
        weights = weights / weights.sum()
        
        # Compute the Wasserstein distance using the POT library
        distance = ot.wasserstein_1d(exp_data, sim_data, v_weights=weights)
        return distance

    def sample_minibatches(self, exp_data, sim_data, sim_accept_reject_data, sim_fPrel_data, 
                           batch_size, K):
        """ Sample K random minibatches from the experimental and simulated data. """
        num_samples = exp_data.size(0)
        indices = torch.randperm(num_samples)[:K * batch_size].view(K, batch_size)
        
        exp_minibatches = []
        sim_minibatches = []
        sim_accept_reject_minibatches = []
        sim_fPrel_minibatches = []

        for idx in indices:
            exp_minibatches.append(exp_data[idx])
            sim_minibatches.append(sim_data[idx])
            sim_accept_reject_minibatches.append(sim_accept_reject_data[idx])
            sim_fPrel_minibatches.append(sim_fPrel_data[idx])

        return exp_minibatches, sim_minibatches, sim_accept_reject_minibatches, sim_fPrel_minibatches

    def train(self, train_loader, val_loader=None, num_epochs=10, verbose=False, 
              n_projections=128, patience=100, delta=0, grad_ave=True, minibatch_size=100, K=50):
        print("Beginning parameter tuning...")
        print(f"Initial (a, b): ({self.weight_nexus.params_a.item():.4f}, {self.weight_nexus.params_b.item():.4f})")
        if self.scheduler is None: 
            print("No learning rate scheduler provided, using constant learning rate:", self.lr)

        best_loss = float('inf')  # Initialize best loss
        epochs_without_improvement = 0  # Counter for early stopping
        best_epoch = 0  # To store the epoch with the best loss

        for epoch in tqdm(range(num_epochs), desc="Wasserstein Tuning", dynamic_ncols=True):
            train_loss = 0.0
            self.weight_nexus.train()
            self.optimizer.zero_grad()  # Zero the gradients before starting the epoch

            # Extract the single batch from the train_loader
            batch = next(iter(train_loader))
            exp_data = batch['exp_data'].detach().to(self.device)
            sim_data = batch['sim_data'].detach().to(self.device)
            sim_accept_reject_data = batch['sim_accept_reject'].detach().to(self.device)
            sim_fPrel_data = batch['sim_fPrel'].detach().to(self.device)
            
            # Sample K minibatches together
            exp_minibatches, sim_minibatches, sim_accept_reject_minibatches, sim_fPrel_minibatches = self.sample_minibatches(
                exp_data, sim_data, sim_accept_reject_data, sim_fPrel_data, minibatch_size, K
            )

            # Compute loss for each minibatch
            for exp_minibatch, sim_minibatch, sim_accept_reject_minibatch, sim_fPrel_minibatch in zip(
                    exp_minibatches, sim_minibatches, sim_accept_reject_minibatches, sim_fPrel_minibatches):
                
                # Calculate weights for the current minibatch
                weights = self.weight_nexus(sim_accept_reject_minibatch, sim_fPrel_minibatch).to(self.device)
                
                # Compute the Wasserstein loss
                loss = self.minibatch_wasserstein_distance(exp_minibatch, sim_minibatch, weights, n_projections=n_projections)
                loss.backward()

            # Average gradients over all minibatches and update weights
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Calculate average training loss for the epoch
            avg_train_loss = sum(loss.item() for _ in range(K)) / K
            self.wass_distances.append(avg_train_loss)

            # Update cumulative minimum Wasserstein loss history
            if len(self.cum_min_wass_distances) == 0:
                self.cum_min_wass_distances.append(avg_train_loss)
            else:
                self.cum_min_wass_distances.append(min(avg_train_loss, self.cum_min_wass_distances[-1]))

            # Check if current loss is the best one and save the parameters if it is
            if avg_train_loss < best_loss - delta:
                best_loss = avg_train_loss
                self.params_best = [self.weight_nexus.params_a.item(), self.weight_nexus.params_b.item()]
                epochs_without_improvement = 0  # Reset patience
                best_epoch = epoch
            else:
                epochs_without_improvement += 1  # Increment counter if no improvement

            # Record current parameter values
            self.param_history.append({
                'epoch': epoch + 1,
                'params_a': self.weight_nexus.params_a.item(),
                'params_b': self.weight_nexus.params_b.item()
            })

            # Step the learning rate scheduler, if one is provided
            if self.scheduler:
                self.scheduler.step()

        return self.weight_nexus
    
    def Wasserstein_flow(self, flow_loader: DataLoader, a_b_init_grid, minibatch_size=32, K=10, n_projections=128):
        """
        Generate gradient flow data using the same loss and dataloader as in training.
        """
        # Initialize gradient tensor
        a_b_gradient = torch.zeros(len(a_b_init_grid), 2, device=self.device)
        loss_grid = torch.zeros(len(a_b_init_grid), device=self.device)
        init_counter = 0

        for a_b_init in tqdm(a_b_init_grid, ncols=100, desc="Wasserstein Flow"):
            # Initialize new weight module with different initial parameters
            self.weight_nexus = LundWeight(self.weight_nexus.params_base, a_b_init, 
                                           over_sample_factor=self.weight_nexus.over_sample_factor).to(self.device)
            self.weight_nexus.eval()  # Set the model to evaluation mode

            a_b_gradient_i = torch.zeros(2, device=self.device)
            loss_over_batches = 0

            for batch in flow_loader:
                # Move batch data to the appropriate device
                exp_data = batch['exp_data'].detach().to(self.device)
                sim_data = batch['sim_data'].detach().to(self.device)
                sim_accept_reject_data = batch['sim_accept_reject'].detach().to(self.device)
                sim_fPrel_data = batch['sim_fPrel'].detach().to(self.device)

                # Sample K minibatches together
                exp_minibatches, sim_minibatches, sim_accept_reject_minibatches, sim_fPrel_minibatches = self.sample_minibatches(
                    exp_data, sim_data, sim_accept_reject_data, sim_fPrel_data, minibatch_size, K
                )

                # Compute loss for each minibatch
                for exp_minibatch, sim_minibatch, sim_accept_reject_minibatch, sim_fPrel_minibatch in zip(
                        exp_minibatches, sim_minibatches, sim_accept_reject_minibatches, sim_fPrel_minibatches):

                    # Calculate weights for the current minibatch
                    weights = self.weight_nexus(sim_accept_reject_minibatch, sim_fPrel_minibatch).to(self.device)

                    # Calculate Wasserstein distance
                    if self.loss == ot.wasserstein_1d:
                        loss = self.loss(exp_minibatch, sim_minibatch, v_weights=weights / torch.sum(weights), require_sort=False)
                    elif self.loss == ot.sliced_wasserstein_distance:
                        loss = self.loss(exp_minibatch, sim_minibatch, a=None, b=weights / torch.sum(weights), 
                                         n_projections=n_projections, p=2, log=False)

                    # Backpropagate to accumulate gradients
                    loss.backward()
                    loss_over_batches += loss.item()

                    # Accumulate gradients
                    a_b_gradient_i[0] -= self.weight_nexus.params_a.grad.clone().detach()
                    a_b_gradient_i[1] -= self.weight_nexus.params_b.grad.clone().detach()

            # Average gradients across batches
            a_b_gradient_i /= len(flow_loader) * K  # Dividing by the number of flow batches and K for average gradient
            loss_over_batches /= len(flow_loader) * K  # Averaging loss over batches

            # Write to the master gradient tensor
            a_b_gradient[init_counter] = a_b_gradient_i.clone()
            loss_grid[init_counter] = loss_over_batches
            init_counter += 1

        # Convert the gradient tensor to a numpy array
        a_b_gradient = a_b_gradient.cpu().numpy()
        loss_grid = loss_grid.cpu().numpy()
        return a_b_gradient, loss_grid

