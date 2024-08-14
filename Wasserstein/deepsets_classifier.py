import torch
import torch.nn as nn
import torch.optim as optim

class DeepSetsClassifier(nn.Module):
    def __init__(self, input_dim, phi_hidden_dim=32, rho_hidden_dim=32):
        super(DeepSetsClassifier, self).__init__()
        
        # Phi network (element-wise processing)
        self.phi = nn.Sequential(
            nn.Linear(input_dim, phi_hidden_dim),
            nn.ReLU(),
            nn.Linear(phi_hidden_dim, phi_hidden_dim),
            nn.ReLU()
        )
        
        # Rho network (permutation-invariant aggregation)
        self.rho = nn.Sequential(
            nn.Linear(phi_hidden_dim, rho_hidden_dim),
            nn.ReLU(),
            nn.Linear(rho_hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, weights=None):
        # x shape: (batch_size, n_particles, input_dim)
        batch_size, n_particles, _ = x.size()
        
        # Apply phi to each particle independently
        x = self.phi(x.view(-1, x.size(-1))).view(batch_size, n_particles, -1)
        
        # Sum aggregation
        x = x.mean(dim=1)
        
        # Apply rho to the aggregated set
        x = self.rho(x)
        
        return x.squeeze()
    
    def train_classifier(self, train_exp_loader, train_sim_loader, val_exp_loader, val_sim_loader, num_epochs=10, learning_rate=0.001):
        criterion = nn.BCELoss()
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

                exp_labels = torch.ones(exp_output.size(0))
                sim_labels = torch.zeros(sim_output.size(0))

                loss_exp = criterion(exp_output, exp_labels)
                loss_sim = criterion(sim_output, sim_labels)
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

                    loss_exp = criterion(exp_output, exp_labels)
                    loss_sim = criterion(sim_output, sim_labels)
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