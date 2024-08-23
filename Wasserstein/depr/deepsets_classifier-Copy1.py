import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

"""def prepare_data(exp_obs, sim_obs, batch_size=32):
    exp_dataset = TensorDataset(exp_obs.clone().detach().requires_grad_(True))
    sim_dataset = TensorDataset(sim_obs.clone().detach().requires_grad_(True))

    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)

    return exp_loader, sim_loader"""

def prepare_data(exp_obs, sim_obs, batch_size=10000):
    # Calculate masks for non-zero entries
    exp_mask = (exp_obs != 0).any(dim=-1)
    sim_mask = (sim_obs != 0).any(dim=-1)
    
    # Create datasets with observations and masks
    exp_dataset = TensorDataset(
        exp_obs.clone().detach().requires_grad_(True),
        exp_mask.clone().detach()
    )
    sim_dataset = TensorDataset(
        sim_obs.clone().detach().requires_grad_(True),
        sim_mask.clone().detach()
    )
    # Create data loaders
    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)
    
    return exp_loader, sim_loader
    
def min_max_scaling(outputs, new_min=-5, new_max=5):
    # Calculate the min and max values of the outputs
    min_val = torch.min(outputs)
    max_val = torch.max(outputs)
    
    # Apply the min-max scaling formula
    scaled_outputs = new_min + (outputs - min_val) / (max_val - min_val) * (new_max - new_min)
    
    return scaled_outputs

def plot_score_histogram(exp_scores, sim_scores, sim_weights=None, same_bins=False, bins=50):
    """
    Plot a histogram of classifier scores for the given data.
    
    Parameters:
    - exp_scores: numpy array or torch.Tensor of scores from the experimental data
    - sim_scores: numpy array or torch.Tensor of scores from the simulated data
    - sim_weights: numpy array or torch.Tensor of weights for the simulated scores, or None
    - same_bins: bool, whether to use the same bins for both histograms
    - bins: int, number of bins for the histogram
    """
    exp_scores = np.array(exp_scores)
    sim_scores = np.array(sim_scores)
    
    if sim_weights is not None:
        sim_weights = np.array(sim_weights)
    
    if same_bins:
        # Compute the bin edges from the combined range
        min_score = min(exp_scores.min(), sim_scores.min())
        max_score = max(exp_scores.max(), sim_scores.max())
        bin_edges = np.linspace(min_score, max_score, bins + 1)
    else:
        bin_edges = bins
    
    plt.figure(figsize=(10, 6))
    
    # Plot histogram for experimental scores
    plt.hist(exp_scores, bins=bin_edges, density=True, alpha=0.7, label="Truth")
    
    # Plot histogram for simulated scores with weights if provided
    if sim_weights is not None:
        plt.hist(sim_scores, bins=bin_edges, density=True, weights=sim_weights, alpha=0.7, label="Base")
    else:
        plt.hist(sim_scores, bins=bin_edges, density=True, alpha=0.7, label="Base")
    
    plt.title('Histogram of Classifier Scores (On Test Set)')
    plt.xlabel('Truth score')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
def mult_classifier_plot(exp_scores, sim_scores):

    unique_exp_scores, exp_counts = np.unique(exp_scores, return_counts=True)
    unique_sim_scores, sim_counts = np.unique(sim_scores, return_counts=True)

    exp_bar_width = np.median(np.diff(unique_exp_scores))
    sim_bar_width = np.median(np.diff(unique_sim_scores))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(unique_exp_scores, exp_counts / np.sum(exp_counts), width=exp_bar_width, alpha=0.7, label='Truth')
    ax.bar(unique_sim_scores, sim_counts / np.sum(sim_counts), width=sim_bar_width, alpha=0.7, label='Base')

    ax.set_xlabel('Scores')
    ax.set_ylabel('Density')
    ax.legend()

    plt.tight_layout()
    plt.show()

class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim, phi_hidden_dim=32, rho_hidden_dim=32,
                 phi_layers=3, rho_layers=3,
                 dropout_prob=0.2, mask_pad=False, momentum=0.1):
        super(DeepSetsClassifier, self).__init__()
        # Inputs:
        #    - mask_pad: whether to apply phi to padding or ignore
        #    - momentum: momentum for batch normalization
        
        self.mask_pad = mask_pad
        s=0.1 # PyTorch default 0.01 works well for multiplicity
        
        # Phi network (element-wise processing)
        phi_layers_list = [nn.Linear(input_dim, phi_hidden_dim),
                           #nn.BatchNorm1d(phi_hidden_dim),
                           nn.LeakyReLU(negative_slope=s),
                           nn.Dropout(dropout_prob)]
        
        for _ in range(1, phi_layers-1):
            phi_layers_list.extend([
                nn.Linear(phi_hidden_dim, phi_hidden_dim),
                #nn.BatchNorm1d(phi_hidden_dim), 
                nn.LeakyReLU(negative_slope=s),
                nn.Dropout(dropout_prob)
            ])
        phi_layers_list.extend([nn.Linear(phi_hidden_dim, phi_hidden_dim),
                nn.LeakyReLU(negative_slope=s)])
        
        self.phi = nn.Sequential(*phi_layers_list)

        # Rho network (permutation-invariant aggregation)
        rho_layers_list = []
        for _ in range(rho_layers - 1):
            rho_layers_list.extend([
                nn.Linear(phi_hidden_dim, rho_hidden_dim),
                # nn.BatchNorm1d(rho_hidden_dim),
                nn.LeakyReLU(negative_slope=s),
                nn.Dropout(dropout_prob)
            ])
            phi_hidden_dim = rho_hidden_dim
        
        rho_layers_list.append(nn.Linear(rho_hidden_dim, 1))
        self.rho = nn.Sequential(*rho_layers_list)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f'Number of learnable parameters: {total_params}')
                    
        self.criterion = nn.BCEWithLogitsLoss()
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, x, mask=None):
        # x shape: (batch_size, n_particles, input_dim)
        # mask shape: (batch_size, n_particles)
        batch_size, n_particles, input_dim = x.size()

        if self.mask_pad:
            # Apply phi only to non-zero entries for efficiency, ignores padding
            if mask is None:
                # Create mask for non-zero entries
                mask = (x != 0).any(dim=-1)  # shape: (batch_size, n_particles)
            # shared whether mask pre-created or not
            non_zero_x = x[mask]  # shape: (num_non_zero, input_dim)
            processed_x = self.phi(non_zero_x)  # shape: (num_non_zero, output_dim)
            output = torch.zeros(batch_size, n_particles, processed_x.size(-1), device=x.device)
            output[mask] = processed_x
            # Aggregate the results
            output = output.sum(dim=1)  # Sum along the particle dimension
        else:
            # Phi applies to all 50 entries
            # For effiiciency we could also apply phi to the pad once and effectively multiply by (50 - mult)
            processed_x = self.phi(x)  # shape: (num_non_zero, output_dim)
            # Create a tensor to hold the processed values
            output = torch.zeros(batch_size, n_particles, processed_x.size(-1), device=x.device)
            # Aggregate by mean
            output = output.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)

        # Apply rho to the aggregated set
        output = self.rho(output)

        return output.squeeze()
    
    def train_classifier(self, train_exp_loader, train_sim_loader, val_exp_loader, val_sim_loader, 
                         device, num_epochs=10, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            
            self.train().to(device)
            train_loss = 0.0
            train_exp_loss = 0.0
            train_sim_loss = 0.0
            
            train_exp_loader_tdqm = train_exp_loader#tqdm(train_exp_loader, desc="Training", leave=False)
            # Training phase
            # optimizer.zero_grad()
            for (exp_batch, sim_batch) in zip(train_exp_loader_tdqm, train_sim_loader):
                
                optimizer.zero_grad()
                
                exp_data = exp_batch[0]
                exp_mask = exp_batch[1]
                sim_data = sim_batch[0]
                sim_mask = sim_batch[1]

                exp_output = self(exp_data, exp_mask)
                sim_output = self(sim_data, sim_mask)

                exp_labels = torch.ones(exp_output.size(0), dtype=exp_output.dtype, device=exp_output.device)
                sim_labels = torch.zeros(sim_output.size(0), dtype=sim_output.dtype, device=sim_output.device)

                loss_exp = self.criterion(exp_output, exp_labels)
                loss_sim = self.criterion(sim_output, sim_labels)
                loss = (loss_exp + loss_sim) / 2

                loss.backward()
                
                optimizer.step()

                train_loss += loss.item() * exp_data.size(0)
                train_exp_loss += loss_exp.item() * exp_data.size(0)
                train_sim_loss += loss_sim.item() * sim_data.size(0) 
                
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_exp_loss = 0.0
            val_sim_loss = 0.0
            average_score = 0.0
            train_sim_loader_tdqm = val_sim_loader#tqdm(val_sim_loader, desc="Validating", leave=False)
            with torch.no_grad():
                for (exp_batch, sim_batch) in zip(train_sim_loader_tdqm, val_sim_loader):
                
                    exp_data = exp_batch[0]
                    exp_mask = exp_batch[1]
                    sim_data = sim_batch[0]
                    sim_mask = sim_batch[1]

                    exp_output = self(exp_data, exp_mask)
                    sim_output = self(sim_data, sim_mask)

                    exp_labels = torch.ones(exp_output.size(0), dtype=exp_output.dtype, device=exp_output.device)
                    sim_labels = torch.zeros(sim_output.size(0), dtype=sim_output.dtype, device=sim_output.device)

                    loss_exp = self.criterion(exp_output, exp_labels)
                    loss_sim = self.criterion(sim_output, sim_labels)
                    loss = (loss_exp + loss_sim) / 2

                    val_loss += loss.item() * exp_data.size(0)
                    val_exp_loss += loss_exp.item() * exp_data.size(0)
                    val_sim_loss += loss_sim.item() * sim_data.size(0)
                    average_score += np.mean(torch.sigmoid(exp_output).cpu().detach().numpy())* exp_data.size(0) + np.mean(torch.sigmoid(sim_output).cpu().detach().numpy())* sim_data.size(0)

            train_loss /= len(train_exp_loader.dataset)
            train_exp_loss /= len(train_exp_loader.dataset)
            train_sim_loss /= len(train_sim_loader.dataset)
            val_loss /= len(val_exp_loader.dataset)
            val_exp_loss /= len(val_exp_loader.dataset)
            val_sim_loss /= len(val_sim_loader.dataset)
            average_score /= len(train_exp_loader.dataset)
            
            
            self.train_loss.append(train_loss)
            self.val_loss.append(val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.5f}, Train Exp Loss: {train_exp_loss:.4f}, Train Sim Loss: {train_sim_loss:.4f}')
            print(f'Val Loss: {val_loss:.5f}, Val Exp Loss: {val_exp_loss:.4f}, Val Sim Loss: {val_sim_loss:.4f}')
            # print(f'Average score: {average_score:.5g}\n')
            
    def evaluate_model(self, test_exp_loader, test_sim_loader):
        self.eval()
        test_loss = 0.0
        with torch.no_grad():
            for exp_batch, sim_batch in zip(test_exp_loader, test_sim_loader):
                exp_data, = exp_batch
                sim_data, = sim_batch

                exp_output = classifier(exp_data)
                sim_output = classifier(sim_data)

                exp_labels = torch.ones(exp_output.size(0))
                sim_labels = torch.zeros(sim_output.size(0))

                loss_exp = self.criterion(exp_output, exp_labels)
                loss_sim = self.criterion(sim_output, sim_labels)
                loss = (loss_exp + loss_sim) / 2

                test_loss += loss.item() * exp_data.size(0)

        print(f'Test Loss: {test_loss/len(test_exp_loader.dataset):.4f}')
        
    def loss_plot(self):

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss, label='Training Loss', marker='o')
        plt.plot(self.val_loss, label='Validation Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()