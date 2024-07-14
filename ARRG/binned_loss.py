import torch
import matplotlib.pyplot as plt

class BinnedLoss(torch.nn.Module):
    def __init__(self, results_dir, print_detials = False):
        super(BinnedLoss, self).__init__()
        self.print_details = print_detials
        self.results_dir = results_dir

    def histogram(self, observable, weights=None, bins=50, min=0.0, max=1.0):
        """
        Generate a differentiable weighted histogram.
        """
        n_samples, n_chns = 1, 1

        # Initialize bins
        hist_torch = torch.zeros(n_samples, n_chns, bins, device=observable.device)
        delta = (max - min) / bins
        bin_table = torch.linspace(min, max, steps=bins, device=observable.device)

        # Perform the binning
        for dim in range(1, bins - 1):
            h_r_sub_1, h_r, h_r_plus_1 = bin_table[dim - 1: dim + 2]

            mask_sub = ((h_r > observable) & (observable >= h_r_sub_1)).float()
            mask_plus = ((h_r_plus_1 > observable) & (observable >= h_r)).float()

            if weights == None:
                hist_torch[:, :, dim].add_(torch.sum(((observable - h_r_sub_1) * mask_sub).view(n_samples, n_chns, -1), dim=-1))
                hist_torch[:, :, dim].add_(torch.sum(((h_r_plus_1 - observable) * mask_plus).view(n_samples, n_chns, -1), dim=-1))
            else:
                hist_torch[:, :, dim].add_(torch.sum(((observable - h_r_sub_1) * mask_sub * weights).view(n_samples, n_chns, -1), dim=-1))
                hist_torch[:, :, dim].add_(torch.sum(((h_r_plus_1 - observable) * mask_plus * weights).view(n_samples, n_chns, -1), dim=-1))

        # Normalize the histogram so that the sum across all bins is 1
        hist_torch = hist_torch.clone() / hist_torch.sum(dim=-1, keepdim=True)

        return (hist_torch / delta).squeeze(), bin_table

    def forward(self, sim_observable, exp_observable, weights):
        """
        Loss function which creates a n-dimensional density estimation 
        and takes the mean-squared-error of an n-dimensional binning.
        """
        # Find min and max of observables
        minimum = torch.min(torch.minimum(sim_observable[:], exp_observable[:]))
        maximum = torch.max(torch.maximum(sim_observable[:], exp_observable[:]))

        # Perform the binning of the macroscopic observables
        histo_sim, bins_sim = self.histogram(sim_observable[:].unsqueeze(0), weights = weights, bins = int(maximum - minimum), min = minimum, max = maximum)
        histo_exp, bins_exp = self.histogram(exp_observable[:].unsqueeze(0), bins = int(maximum - minimum), min = minimum, max = maximum)
        #histo_sim, bins_sim = self.differentiable_histogram(sim_observable[:], weights = weights, bins = int(maximum - minimum), min = minimum, max = maximum)
        #histo_exp, bins_exp = self.differentiable_histogram(exp_observable[:], bins = int(maximum - minimum), min = minimum, max = maximum)

        # Compute the psuedo-chi^2

        # TBD insert stochastic uncertainty into the denominator of the loss with 1% lower bound as done in 1610.08328 
        #error = torch.ones(len(bins_exp))
        #error[(histo_exp > 0.) | (histo_sim > 0.)] = torch.div(error, (histo_exp + histo_sim))

        pseudo_chi2 = (torch.pow((histo_sim - histo_exp), 2))

        if self.print_details:
            # Bin the simualted observable
            histo_sim_OG, bins_sim_OG = self.histogram(sim_observable[:].unsqueeze(0), bins = int(maximum - minimum), min = minimum, max = maximum)
            histo_sim_OG, bins_sim_OG = self.histogram(sim_observable[:], bins = int(maximum - minimum), min = minimum, max = maximum)
            # Plot historgrams to ensure reweighting is working as expected
            fig, ax = plt.subplots(1,1,figsize = (6,5))
            ax.plot(bins_sim.detach().numpy(), histo_sim.detach().numpy(), '-o', label = 'Weighted')#label = r'$\mathrm{Weighted}$')
            ax.plot(bins_exp.detach().numpy(), histo_exp.detach().numpy(), '-o', label = 'Exp.')#label = r'$\mathrm{Exp.}$')
            ax.plot(bins_sim_OG.detach().numpy(), histo_sim_OG.detach().numpy(), '-o', label = 'Sim.')#label = r'$\mathrm{Sim.}$')
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