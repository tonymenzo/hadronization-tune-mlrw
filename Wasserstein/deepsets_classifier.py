import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np

def prepare_data(exp_obs, sim_obs, batch_size=32):
    exp_dataset = TensorDataset(torch.tensor(exp_obs, dtype=torch.float32))
    sim_dataset = TensorDataset(torch.tensor(sim_obs, dtype=torch.float32))

    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=True)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=True)

    return exp_loader, sim_loader
    
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

class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim, phi_hidden_dim=32, rho_hidden_dim=32, dropout_prob=0.2, phi_first_layer_bias=True):
        super(DeepSetsClassifier, self).__init__()
        
        # Phi network (element-wise processing)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden_dim),
            nn.BatchNorm1d(phi_hidden_dim),  # Batch normalization after the first linear layer
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.BatchNorm1d(phi_hidden_dim),  # Batch normalization after the second linear layer
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.BatchNorm1d(phi_hidden_dim),  # Batch normalization after the second linear layer
            nn.LeakyReLU(),
        )
        
        # Rho network (permutation-invariant aggregation)
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden_dim, rho_hidden_dim),
            nn.BatchNorm1d(rho_hidden_dim),  # Batch normalization after the first linear layer
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(rho_hidden_dim, rho_hidden_dim),
            nn.BatchNorm1d(phi_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(rho_hidden_dim, rho_hidden_dim),
            nn.BatchNorm1d(phi_hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_prob),
            
            nn.Linear(rho_hidden_dim, 1),
            #nn.Sigmoid(),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
        self.criterion = nn.BCEWithLogitsLoss()
    
    def forward(self, x, weights=None):
        # x shape: (batch_size, n_particles, input_dim)
        batch_size, n_particles, input_dim = x.size()

        # Create a mask for non-padded entries
        mask = (x != 0).any(dim=-1)  # shape: (batch_size, n_particles)

        # Apply phi to each particle independently
        x = self.phi(x.view(-1, input_dim)).view(batch_size, n_particles, -1)

        # Apply the mask to x
        x = x * mask.unsqueeze(-1)

        # Aggregation with masked mean
        x = x.sum(dim=1)  # Sum along the particle dimension
        
        # Apply rho to the aggregated set
        x = self.rho(x)
        
        return x.squeeze()
    
    def train_classifier(self, train_exp_loader, train_sim_loader, val_exp_loader, val_sim_loader, num_epochs=10, learning_rate=0.001):
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0.0
            train_exp_loss = 0.0
            train_sim_loss = 0.0

            # Training phase
            for exp_batch, sim_batch in zip(train_exp_loader, train_sim_loader):
                optimizer.zero_grad()
                exp_data = exp_batch[0]
                sim_data = sim_batch[0]

                exp_output = self(exp_data)
                sim_output = self(sim_data)

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
            with torch.no_grad():
                for exp_batch, sim_batch in zip(val_exp_loader, val_sim_loader):
                    exp_data = exp_batch[0]
                    sim_data = sim_batch[0]

                    exp_output = self(exp_data)
                    sim_output = self(sim_data)

                    exp_labels = torch.ones(exp_output.size(0))
                    sim_labels = torch.zeros(sim_output.size(0))

                    loss_exp = self.criterion(exp_output, exp_labels)
                    loss_sim = self.criterion(sim_output, sim_labels)
                    loss = (loss_exp + loss_sim) / 2

                    val_loss += loss.item() * exp_data.size(0)
                    val_exp_loss += loss_exp.item() * exp_data.size(0)
                    val_sim_loss += loss_sim.item() * sim_data.size(0)

            train_loss /= len(train_exp_loader.dataset)
            train_exp_loss /= len(train_exp_loader.dataset)
            train_sim_loss /= len(train_sim_loader.dataset)
            val_loss /= len(val_exp_loader.dataset)
            val_exp_loss /= len(val_exp_loader.dataset)
            val_sim_loss /= len(val_sim_loader.dataset)

            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'Train Loss: {train_loss:.4f}, Train Exp Loss: {train_exp_loss:.4f}, Train Sim Loss: {train_sim_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Exp Loss: {val_exp_loss:.4f}, Val Sim Loss: {val_sim_loss:.4f}\n')
            
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