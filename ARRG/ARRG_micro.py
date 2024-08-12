import numpy as np
from tqdm import tqdm

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_style("ticks")
#sns.set_context("paper", font_scale = 1.8)
#plt.rcParams['text.usetex'] = True
#plt.rcParams['font.size'] = 25

from lund_weight import LundWeight
#from pseudo_chi2_loss import PseudoChiSquareLoss
#from wasserstein_1d_macro_loss import WassersteinLoss1D
from wasserstein_micro_loss import WassersteinLossMicro

class ARRG():
    def __init__(self, epochs, dim_multiplicity, dim_accept_reject, over_sample_factor, params_base,
                 sim_observable_dataloader, sim_z_dataloader, sim_fPrel_dataloader,
                 exp_observable_dataloader, params_init = None, print_details = False, results_dir = None, fixed_binning = True):
        """
        ARRG training class for tuning microscopic dynamics (hadronization parameters) from macroscopic observables.

        epochs (int): ---------------------- Number of training cycles through the full dataset
        dim_multiplicity (int): ------------ Size of the zero-padded hadronization data (largest fragmentation chain length)
        dim_accept_reject (int): ----------- Size of zero-padded accept-reject arrays (largest rejection count in events)
        over_sample_factor (float): --------
        params_base (torch tensor): -------- 
        sim_observable_dataloader (string): - Path to global observable data generated by sim model
        sim_z_dataloader (string): ---------- Path to sim model kinematics generating the global observable
        exp_observable_dataloader (string): - Path to desired (experimental) global distribution (training data)
        params_init (torch tensor): ---------
        print_details (bool): --------------- Option to output intermediate results during training
        results_dir (string): --------------- Option for path to store results (if the directory doesn't exist, it will be created)
        """

        # Model hyperparameters
        self.epochs = epochs
        self.dim_multiplicity = dim_multiplicity 
        self.dim_accept_reject = dim_accept_reject
        self.over_sample_factor = over_sample_factor
        self.params_base = params_base
        if params_init == None:
            self.params_init = self.params_base
        else:
            self.params_init = params_init

        # Training data
        self.exp_observable = exp_observable_dataloader
        self.sim_observable_base = sim_observable_dataloader
        self.sim_z_base = sim_z_dataloader
        self.sim_fPrel_base = sim_fPrel_dataloader
        self.print_details = print_details
        self.results_dir = results_dir
        self.fixed_binning = fixed_binning

        # Initialize the Lund weight module
        self.weight_nexus = LundWeight(self.params_base, self.params_init, over_sample_factor = self.over_sample_factor)
        
        # Initialize the loss
        #self.pseudo_chi2_loss = PseudoChiSquareLoss(results_dir = self.results_dir , print_detials = self.print_details, fixed_binning = self.fixed_binning)
        #self.wasserstein_loss = WassersteinLoss1D(p = 1)
        self.wasserstein_loss = WassersteinLossMicro()

        # Create a results directory if it doesn't exist
        if self.results_dir != None:
            if not os.path.exists(self.results_dir):
                os.mkdir(self.results_dir)
                print('A model directory was created at,', self.results_dir)

    def train_ARRG(self, optimizer, scheduler=None):
        """
        Training cycle for ARRG. 

        optimizer: Specified network optimizer
        scheduler: Specified learning rate scheduler
        """
        a_b = [np.array([self.weight_nexus.params_a.clone().detach().numpy(), self.weight_nexus.params_b.clone().detach().numpy()])]
        #loss_epoch = []
        batch_counter = 0
        # TBD: Optimization for GPU training
        for i in tqdm(range(self.epochs), ncols = 100):
            device = "cpu"
            batch_counter = 0
            for (x,y,z,w) in zip(self.sim_z_base, self.sim_mT_base, self.sim_observable_base, self.exp_observable):
                print('Batch #', batch_counter)
                x, y, z, w = x.to(device), y.to(device), z.to(device), w.to(device)
                # Reset the gradients in the optimizer
                optimizer.zero_grad()
                # Compute the weights
                weights = self.weight_nexus(x, y, z)
                # Compute the loss
                loss = self.pseudo_chi2_loss(z, w, weights) #/ x.shape[0]
                print('----------------------------------------------')
                print('Loss:', loss.clone().detach().numpy())
                # Compute gradients via backprop
                loss.backward()
                # Output the gradients of a and b
                switch = 0
                for param in self.weight_nexus.parameters():
                    if switch == 0:
                        switch += 1
                        print('Gradient of a:', param.grad.clone().detach().numpy())
                    else:
                        print('Gradient of b:', param.grad.clone().detach().numpy())
                # Update the network weights
                optimizer.step()
                # Update the learning rate scheduler
                scheduler.step(loss)
                # Iterate the batch counter
                batch_counter+=1
                """
                # Constrain weight layer parameters within allowed Pythia range
                switch = 0
                for p in self.weight_nexus.parameters():
                    if switch == 0:
                        switch += 1
                        p.data.clamp_(0.0, 2.0)
                    else:
                        p.data.clamp_(0.2, 2.0)
                """

                # Output the loss and learning rate 
                print(f'Loss: {loss.clone().detach().numpy():>8f}, LR: {optimizer.param_groups[0]["lr"]:>8f}')
                print(f'a: {self.weight_nexus.params_a.clone().detach().numpy()}, b: {self.weight_nexus.params_b.clone().detach().numpy()}')
                print('----------------------------------------------')

                # Record the epoch loss
                #loss_epoch.append(loss.detach().numpy())

                # Record the tuned parameters
                a_b.append(np.array([self.weight_nexus.params_a.clone().detach().numpy(), self.weight_nexus.params_b.clone().detach().numpy()]))
                
                if self.print_details:
                    # Check the histograms
                    _, bins_exp = np.histogram(w[:].detach().numpy())
                    _, bins_sim = np.histogram(z[:].detach().numpy())
                    _, bins_fine_tuned = np.histogram(z[:].detach().numpy(), weights = weights.detach().numpy())
    
                    min_exp, max_exp = bins_exp[0], bins_exp[-1]
                    min_sim, max_sim = bins_sim[0], bins_sim[-1]
                    min_fine_tuned, max_fine_tuned = bins_fine_tuned[0], bins_fine_tuned[-1]
                    
                    # Plot multiplicity
                    fig_1, ax_1 = plt.subplots(1,1,figsize=(6,5))
                    ax_1.hist(z[:].detach().numpy(), int(max_sim - min_sim), alpha = 0.5, density = True, edgecolor = 'black', label = 'Base')#label = r'$\mathrm{Base}$')
                    ax_1.hist(w[:].detach().numpy(), int(max_exp - min_exp), alpha = 0.5, density = True, edgecolor = 'black', label = 'Exp.')#label = r'$\mathrm{Exp.}$')
                    ax_1.hist(z[:].detach().numpy(), int(max_fine_tuned - min_fine_tuned), weights = weights.detach().numpy(), alpha = 0.5, density = True, edgecolor = 'black', label = 'Tuned')#label = r'$\mathrm{Tuned}$')
                    ax_1.set_xlabel(r'$N_h$')
                    ax_1.set_ylabel(r'$\mathrm{Count}$')
                    ax_1.legend(frameon=False)
                    fig_1.tight_layout()
                    fig_1.savefig(self.results_dir + r'/ARRG_multiplicity_base_vs_exp_vs_tuned.pdf', dpi=300, pad_inches = .1, bbox_inches = 'tight')
    
                    # Plot the search space
                    a_b_target = np.array([0.68, 0.98]) # Monash
                    fig_2, ax_2 = plt.subplots(1,1,figsize=(6,5))
                    ax_2.plot(np.array(a_b)[:,0], np.array(a_b)[:,1], 'o-', ms = 1.5, alpha = 1.0, color = 'blue')
                    ax_2.plot(a_b_target[0], a_b_target[1], 'x', color='green', label = 'Target')#label = r'$\mathrm{Target}$')
                    ax_2.plot(self.params_init[0].clone().detach().numpy(), self.params_init[1].clone().detach().numpy(), 'x', color = 'red', label = 'Initial')#label = r'$\mathrm{Initial}$')
                    # Set relevant axis limits
                    ax_2.set_xlim(a_b_target[0]-0.1, self.params_base[0].clone().detach().numpy()+0.1)
                    ax_2.set_ylim(self.params_base[0].clone().detach().numpy()-0.1, a_b_target[1]+0.1)
                    ax_2.set_xlabel(r'$a$')
                    ax_2.set_ylabel(r'$b$')
                    ax_2.legend(frameon = False, loc = 'upper right')
                    fig_2.tight_layout()
                    fig_2.savefig(self.results_dir + r'/ARRG_search_space.pdf', dpi=300, pad_inches = .1, bbox_inches = 'tight')
                    
                    # Close figures so RAM isn't soaked up
                    plt.close(fig_1)
                    plt.close(fig_2)
                
        return np.array([self.weight_nexus.params_a.clone().detach().numpy(), self.weight_nexus.params_b.clone().detach().numpy()]), a_b
    
    def ARRG_flow(self, optimizer, a_b_init_grid):
        """
        Generate gradient flow and loss landscape data
        """
        # Initialize gradient tensor
        a_b_gradient = torch.zeros(len(a_b_init_grid), 2)
        loss_grid = torch.zeros(len(a_b_init_grid))
        device = 'cpu'
        init_counter = 0
        for a_b_init in tqdm(a_b_init_grid, ncols=100):
            #tqdm.write(f"a: {a_b_init[0]}, b: {a_b_init[1]}")
            # Create an intermediate gradient tensor
            a_b_gradient_i = torch.zeros(2)
            # Initialize new weight module with different initial parameters
            self.weight_nexus = LundWeight(self.params_base, a_b_init, over_sample_factor = self.over_sample_factor)
            for (x,y,z,w) in zip(self.sim_z_base, self.sim_fPrel_base, self.sim_observable_base, self.exp_observable):
                x, y, z, w = x.to(device), y.to(device), z.to(device), w.to(device)
                # Reset the gradients
                optimizer.zero_grad()
                # Compute the weights
                weights = self.weight_nexus(x, y)
                # Compute the loss
                #loss = self.pseudo_chi2_loss(z, w, weights) / x.shape[0]
                loss = self.wasserstein_loss(z, w, weights)
                
                # Compute gradients via backprop
                loss.backward()
                # Save the gradients of a and b
                switch = 0
                print('----------------------------------------------')
                #print('a:', self.weight_nexus.parameters()[0], 'b:', self.weight_nexus.parameters()[1])
                print('Loss:', loss.clone().detach().numpy())
                for param in self.weight_nexus.parameters():
                    if switch == 0:
                        switch += 1
                        print('a:', param.clone().detach().numpy())
                        print('Gradient of a:', param.grad.clone().detach().numpy())
                        a_b_gradient_i[0] = param.grad.clone().detach()
                    else:
                        print('b:', param.clone().detach().numpy())
                        print('Gradient of b:', param.grad.clone().detach().numpy())
                        a_b_gradient_i[1] = param.grad.clone().detach()
                print('----------------------------------------------')
            # Write to the master gradient tensor
            a_b_gradient[init_counter] = a_b_gradient_i.clone()
            # Write to the master loss tensor
            loss_grid[init_counter] = loss.clone().detach()
            # Iterate the init_counter
            init_counter += 1
        # Convert the gradient and loss tensors to numpy arrays
        a_b_gradient = a_b_gradient.numpy()
        loss_grid = loss_grid.numpy()
        return a_b_gradient, loss_grid