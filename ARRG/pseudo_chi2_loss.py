import torch
import matplotlib.pyplot as plt
import numpy as np

class PseudoChiSquareLoss(torch.nn.Module):
    def __init__(self, results_dir, print_detials = False, fixed_binning = True):
        super(PseudoChiSquareLoss, self).__init__()
        self.print_details = print_detials
        self.results_dir = results_dir
        self.fixed_binning = fixed_binning

        if self.fixed_binning:
            # Load a fixed binning for the pseudo-chi^2 loss
            self.bins = torch.tensor(np.load('monash_binning.npy'))
            # Set in stone the bin counts and uncertainty for the 'experimental' dataset using large statistics
            self.histo_exp = torch.tensor(np.load('monash_counts.npy'))
            self.histo_exp_norm = self.histo_exp / torch.sum(self.histo_exp)

    def histogram(self, observable, weights=None, bins=None, min=0.0, max=1.0):
        """
        Generate a differentiable weighted histogram.
        """
        n_samples, n_chns = 1, 1

        # Initialize bins
        if bins is not None and self.fixed_binning:
            bin_table = bins.to(observable.device)
            bins = len(bin_table) - 1
            min = bin_table[0]
            max = bin_table[-1]
        else:
            bin_table = torch.linspace(min, max, steps=bins + 1, device=observable.device)

        hist_torch = torch.zeros(n_samples, n_chns, bins, device=observable.device)
        squared_weight_sums = torch.zeros(n_samples, n_chns, bins, device=observable.device)
        delta = (max - min) / bins

        # If weights are None, use uniform weights
        if weights is None:
            weights = torch.ones_like(observable)

        # Perform the binning
        for dim in range(1, bins - 1):
            h_r_sub_1, h_r, h_r_plus_1 = bin_table[dim - 1: dim + 2]

            mask_sub = ((h_r > observable) & (observable >= h_r_sub_1)).float()
            mask_plus = ((h_r_plus_1 > observable) & (observable >= h_r)).float()

            sub_weights = (observable - h_r_sub_1) / delta * weights * mask_sub
            plus_weights = (h_r_plus_1 - observable) / delta * weights * mask_plus

            hist_torch[:, :, dim].add_(torch.sum(sub_weights.view(n_samples, n_chns, -1), dim=-1))
            hist_torch[:, :, dim].add_(torch.sum(plus_weights.view(n_samples, n_chns, -1), dim=-1))

            squared_weight_sums[:, :, dim].add_(torch.sum((sub_weights**2).view(n_samples, n_chns, -1), dim=-1))
            squared_weight_sums[:, :, dim].add_(torch.sum((plus_weights**2).view(n_samples, n_chns, -1), dim=-1))

        return hist_torch.squeeze(), bin_table, squared_weight_sums.squeeze()

    def forward(self, sim_observable, exp_observable, weights):
        """
        Loss function which creates a n-dimensional density estimation 
        and takes the mean-squared-error of an n-dimensional binning.
        """
        if not self.fixed_binning:
            # Find min and max of observables
            minimum = torch.min(torch.minimum(sim_observable[:], exp_observable[:]))
            maximum = torch.max(torch.maximum(sim_observable[:], exp_observable[:]))

            # Perform the binning of the macroscopic observables
            histo_sim, bins_sim, weight_sum_sq_sim = self.histogram(sim_observable[:].unsqueeze(0), weights = weights, bins = int(maximum - minimum), min = minimum, max = maximum)
            histo_exp, bins_exp, _ = self.histogram(exp_observable[:].unsqueeze(0), bins = int(maximum - minimum), min = minimum, max = maximum)
            #histo_sim, bins_sim = self.differentiable_histogram(sim_observable[:], weights = weights, bins = int(maximum - minimum), min = minimum, max = maximum)
            #histo_exp, bins_exp = self.differentiable_histogram(exp_observable[:], bins = int(maximum - minimum), min = minimum, max = maximum)

            # Compute the psuedo-chi^2
            # For bins with zero counts, add a small constant to avoid division by zero
            epsilon = 1e-10
            # The uncertainty on a weighted bin is given by sig^2 = sum_i (w_i^2) where w_i represents all weights in the given bin.
            # For a normalized distribution the weighted/unweighted uncertainty is normalized by the 'area' of the distribution squared.
            uncertainty_sim = weight_sum_sq_sim / torch.pow((torch.sum(histo_sim) + epsilon), 2)
            uncertainty_exp = (histo_exp * (1 - histo_exp / torch.sum(histo_exp))) / torch.pow((torch.sum(histo_exp) + epsilon), 2) # Poisson uncertainty

            # Ensure a minimum uncertainty of 1% of the bin content
            #min_uncertainty_sim = 0.01 * histo_sim
            #min_uncertainty_exp = 0.01 * histo_exp

            #uncertainty_sim = torch.maximum(uncertainty_sim, min_uncertainty_sim) + epsilon
            #uncertainty_exp = torch.maximum(uncertainty_exp, min_uncertainty_exp) + epsilon

            # Normalize the histograms
            histo_sim = histo_sim / torch.sum(histo_sim)
            histo_exp = histo_exp / torch.sum(histo_exp)

            # Compute the chi-squared statistic
            pseudo_chi2 = torch.pow((histo_sim - histo_exp), 2) / (uncertainty_sim + uncertainty_exp)
        else:
            # Perform the binning of the macroscopic observables
            histo_sim, bins_sim, weight_sum_sq_sim = self.histogram(sim_observable[:].unsqueeze(0), weights = weights, bins = self.bins)
            # Compute the psuedo-chi^2
            # For bins with zero counts, add a small constant to avoid division by zero
            epsilon = 1e-10
            # The uncertainty on a weighted bin is given by sig^2 = sum_i (w_i^2) where w_i represents all weights in the given bin.
            # For a normalized distribution the weighted/unweighted uncertainty is normalized by the 'area' of the distribution squared.
            uncertainty_sim = weight_sum_sq_sim / torch.pow((torch.sum(histo_sim) + epsilon), 2) + epsilon
            uncertainty_exp = (self.histo_exp * (1 - self.histo_exp / torch.sum(self.histo_exp))) / torch.pow((torch.sum(self.histo_exp) + epsilon), 2) + epsilon # Poisson uncertainty

            #print('sigma_sim',uncertainty_sim)
            #print('sigma_exp',uncertainty_exp)
            
            # Normalize the simulated histogram
            histo_sim = histo_sim / torch.sum(histo_sim)

            # Compute the chi-squared statistic
            pseudo_chi2 = torch.pow((histo_sim - self.histo_exp_norm), 2) / (uncertainty_sim + uncertainty_exp)

        if self.print_details:
            # Bin the simualted observable
            if self.fixed_binning:
                histo_sim_OG, bins_sim_OG, _ = self.histogram(sim_observable[:].unsqueeze(0), bins = self.bins)
            else:
                histo_sim_OG, bins_sim_OG, _, _ = self.histogram(sim_observable[:].unsqueeze(0), bins = int(maximum - minimum), min = minimum, max = maximum)
                #histo_sim_OG, bins_sim_OG, _, _ = self.histogram(sim_observable[:], bins = int(maximum - minimum), min = minimum, max = maximum)
            
            # Normalize the base histogram
            histo_sim_OG = histo_sim_OG / torch.sum(histo_sim_OG)
            
            # Compute the error bars
            error_bars_sim = torch.sqrt(uncertainty_sim)
            error_bars_exp = torch.sqrt(uncertainty_exp)
            # Plot historgrams to ensure reweighting is working as expected
            fig, ax = plt.subplots(1,1,figsize = (6,5))
            ax.plot(bins_sim.detach().numpy()[0:-1], histo_sim.detach().numpy(), '-o', label = 'Weighted', color = 'tab:blue')#label = r'$\mathrm{Weighted}$')
            if self.fixed_binning:
                ax.plot(self.bins.detach().numpy()[0:-1], self.histo_exp_norm.detach().numpy(), '-o', label = 'Exp.', color = 'tab:orange')
                ax.errorbar(self.bins.detach().numpy()[0:-1], self.histo_exp_norm.detach().numpy(), yerr=error_bars_exp.detach().numpy(), fmt='none', color='tab:orange', capsize=2)
            else:
                ax.plot(bins_exp.detach().numpy(), histo_exp.detach().numpy(), '-o', label = 'Exp.', color = 'tab:orange')#label = r'$\mathrm{Exp.}$')
                ax.errorbar(bins_exp.detach().numpy(), histo_exp.detach().numpy(), yerr=error_bars_exp.detach().numpy(), fmt='none', color='tab:orange', capsize=2)
            ax.plot(bins_sim_OG.detach().numpy()[0:-1], histo_sim_OG.detach().numpy(), '-o', label = 'Sim.', color = 'tab:green')#label = r'$\mathrm{Sim.}$')

            # Plot the error bars
            ax.errorbar(bins_sim.detach().numpy()[0:-1], histo_sim.detach().numpy(), yerr=error_bars_sim.detach().numpy(), fmt='none', color='tab:blue', capsize=2)

            ax.legend(frameon = False)
            fig.tight_layout()
            fig.savefig(self.results_dir + r'/loss_binning_check.pdf', dpi=300, pad_inches = .1, bbox_inches = 'tight')
            plt.close(fig)

        return torch.sum(pseudo_chi2)

class BinnedLoss_v2(torch.nn.Module):
    def __init__(self, print_details=False):
        super(BinnedLoss_v2, self).__init__()
        self.print_details = print_details

    def differentiable_histogram(self, x, weights=None, bins=50, min=0.0, max=1.0):
        if x.dim() != 1:
            raise ValueError('Input tensor must be 1-dimensional.')

        if min is None:
            min = x.min().item()
        if max is None:
            max = x.max().item()

        if weights is None:
            weights = torch.ones_like(x)
        elif weights.shape != x.shape:
            raise ValueError('Weights must have the same shape as the input tensor.')

        delta = (max - min) / bins
        bin_edges = torch.linspace(min, max, bins + 1, device=x.device)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        x = x.unsqueeze(1)  # Shape: (n_elements, 1)
        weights = weights.unsqueeze(1)  # Shape: (n_elements, 1)

        # Compute distances to bin centers
        diff = x - centers

        # Compute weights for each bin
        weight_left = torch.nn.functional.relu(1 - torch.abs(diff) / delta)
        weight_right = torch.nn.functional.relu(1 - torch.abs(diff - delta) / delta)

        # Combine weights and compute histogram
        hist = torch.sum(weights * (weight_left + weight_right), dim=0)

        # Normalize the histogram to sum to 1
        hist = hist / torch.sum(hist)

        return hist, bin_edges

    def forward(self, sim_observable, exp_observable, weights=None, print_details = False):
        """
        Loss function which creates a one-dimensional density estimation 
        and takes the mean-squared-error of a one-dimensional binning.
        """
        # Ensure inputs are 1D tensors
        sim_observable = sim_observable.flatten()
        exp_observable = exp_observable.flatten()
        if weights is not None:
            weights = weights.flatten()

        # Find min and max of observables
        minimum = torch.min(torch.min(sim_observable), torch.min(exp_observable))
        maximum = torch.max(torch.max(sim_observable), torch.max(exp_observable))

        # Determine number of bins
        num_bins = int(maximum - minimum) + 1  # +1 to ensure we cover the entire range

        # Perform the binning of the macroscopic observables
        histo_sim, bins_sim = self.differentiable_histogram(sim_observable, weights=weights, 
                                                            bins=num_bins, min=minimum, max=maximum)
        histo_exp, bins_exp = self.differentiable_histogram(exp_observable, 
                                                            bins=num_bins, min=minimum, max=maximum)

        # Compute the pseudo-chi^2
        pseudo_chi2 = torch.pow((histo_sim - histo_exp), 2)

        if print_details:
            # Plot histograms to ensure reweighting is working as expected
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            bin_centers = (bins_sim[:-1] + bins_sim[1:]) / 2
            ax.plot(bin_centers.detach().numpy(), histo_sim.detach().numpy(), '-o', label='Weighted')
            ax.plot(bin_centers.detach().numpy(), histo_exp.detach().numpy(), '-o', label='Exp.')

            # If you want to plot the unweighted simulation histogram:
            histo_sim_unweighted, _ = self.differentiable_histogram(sim_observable, 
                                                                    bins=num_bins, min=minimum, max=maximum)
            ax.plot(bin_centers.detach().numpy(), histo_sim_unweighted.detach().numpy(), '-o', label='Sim. (Unweighted)')

            ax.legend(frameon=False)
            ax.set_xlabel('Observable')
            ax.set_ylabel('Count')
            fig.tight_layout()
            fig.savefig(self.results_dir + '/loss_binning_check.pdf', dpi=300, pad_inches=.1, bbox_inches='tight')
            plt.close(fig)

        return torch.sum(pseudo_chi2)