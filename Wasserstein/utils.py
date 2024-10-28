import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import numpy as np

def prepare_data(exp_obs, sim_obs, batch_size=10000, num_workers=4, pin_memory=True):
    # Calculate masks for non-zero entries
    exp_mask = (exp_obs != 0).any(dim=-1)
    sim_mask = (sim_obs != 0).any(dim=-1)
    
    # Create datasets with observations and masks
    exp_dataset = TensorDataset(
        exp_obs.clone().detach().requires_grad_(False),
        exp_mask.clone().detach()
    )
    sim_dataset = TensorDataset(
        sim_obs.clone().detach().requires_grad_(False),
        sim_mask.clone().detach()
    )
    # Create data loaders
    exp_loader = DataLoader(exp_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    sim_loader = DataLoader(sim_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=pin_memory)
    
    return exp_loader, sim_loader
    
def min_max_scaling(outputs, new_min=-5, new_max=5):
    # Calculate the min and max values of the outputs
    min_val = torch.min(outputs)
    max_val = torch.max(outputs)
    
    # Apply the min-max scaling formula
    scaled_outputs = new_min + (outputs - min_val) / (max_val - min_val) * (new_max - new_min)
    
    return scaled_outputs

def prescale(exp_data, sim_data):
    # Check if inputs are NumPy arrays or PyTorch tensors
    if isinstance(exp_data, np.ndarray) and isinstance(sim_data, np.ndarray):
        backend = np
        to_float = lambda x: x.astype(float)  # For NumPy, use astype(float)
    elif isinstance(exp_data, torch.Tensor) and isinstance(sim_data, torch.Tensor):
        backend = torch
        to_float = lambda x: x.float()  # For PyTorch, use float() method
    else:
        raise ValueError("Both inputs must be either NumPy arrays or PyTorch tensors")

    # Check if inputs are 1D, 2D, or 3D
    if exp_data.ndim not in [1, 2, 3] or sim_data.ndim not in [1, 2, 3]:
        raise ValueError("Inputs must be 1D, 2D, or 3D arrays or tensors")
    
    if exp_data.ndim != sim_data.ndim:
        raise ValueError("Both inputs must have the same number of dimensions")

    # Create masks for non-padded entries (for 2D or 3D cases)
    if exp_data.ndim == 3:
        non_padded_mask_exp = ~backend.all(exp_data == 0, axis=-1)
        non_padded_mask_sim = ~backend.all(sim_data == 0, axis=-1)
    elif exp_data.ndim == 2:
        non_padded_mask_exp = backend.any(exp_data != 0, axis=-1)
        non_padded_mask_sim = backend.any(sim_data != 0, axis=-1)

    # Combine non-padded data for mean and std calculation
    if exp_data.ndim == 3 or exp_data.ndim == 2:
        combined_data = backend.concatenate([exp_data[non_padded_mask_exp], sim_data[non_padded_mask_sim]], axis=0)
    else:  # For 1D data
        combined_data = backend.concatenate([exp_data, sim_data], axis=0)

    # Cast the combined data to float for mean and std calculation
    combined_data = to_float(combined_data)

    # Calculate combined mean and std
    combined_mean = backend.mean(combined_data, axis=0)
    combined_std = backend.std(combined_data, axis=0)

    print("Mean:", combined_mean)
    print("Std:", combined_std)

    # Scale the data
    exp_data_scaled = backend.clone(exp_data) if backend is torch else backend.copy(exp_data)
    sim_data_scaled = backend.clone(sim_data) if backend is torch else backend.copy(sim_data)

    if exp_data.ndim == 3:
        exp_data_scaled[non_padded_mask_exp] = (to_float(exp_data[non_padded_mask_exp]) - combined_mean[backend.newaxis, :]) / combined_std[backend.newaxis, :]
        sim_data_scaled[non_padded_mask_sim] = (to_float(sim_data[non_padded_mask_sim]) - combined_mean[backend.newaxis, :]) / combined_std[backend.newaxis, :]
    elif exp_data.ndim == 2:
        exp_data_scaled[non_padded_mask_exp] = (to_float(exp_data[non_padded_mask_exp]) - combined_mean) / combined_std
        sim_data_scaled[non_padded_mask_sim] = (to_float(sim_data[non_padded_mask_sim]) - combined_mean) / combined_std
    else:  # For 1D data
        exp_data_scaled = (to_float(exp_data) - combined_mean) / combined_std
        sim_data_scaled = (to_float(sim_data) - combined_mean) / combined_std

    return exp_data_scaled, sim_data_scaled

def presort_data(exp_data, sim_data, sim_fPrel, sim_accept_reject):
    if exp_data.dim() == 1:
        # Sort exp_data independently
        sorted_exp_scores, _ = torch.sort(exp_data)

        # Sort sim_data and get sorting indices
        sorted_sim_scores, sim_sort_indices = torch.sort(sim_data)

        # Use sim_sort_indices to sort sim_fPrel and sim_accept_reject
        sorted_sim_fPrel = sim_fPrel[sim_sort_indices]
        sorted_sim_accept_reject = sim_accept_reject[sim_sort_indices]

        return sorted_exp_scores, sorted_sim_scores, sorted_sim_fPrel, sorted_sim_accept_reject
    
    else:
        # In the case of multiple dimensions, no sort
        return exp_data, sim_data, sim_fPrel, sim_accept_reject
    
class CombinedDataset(Dataset):
    def __init__(self, exp_obs, sim_obs, sim_fPrel, sim_accept_reject, pre_sorted=False):
        if not pre_sorted:
            self.exp_obs, self.sim_obs, self.sim_fPrel, self.sim_accept_reject = presort_data(
                exp_obs, sim_obs, sim_fPrel, sim_accept_reject
            )
        else:
            self.exp_obs = exp_obs
            self.sim_obs = sim_obs
            self.sim_fPrel = sim_fPrel
            self.sim_accept_reject = sim_accept_reject
        
        #assert len(self.exp_obs) == len(self.sim_obs) == len(self.sim_fPrel) == len(self.sim_accept_reject), "All inputs must have the same length"

    def __len__(self):
        return len(self.exp_obs)

    def __getitem__(self, idx):
        return {
            'exp_data': self.exp_obs[idx],
            'sim_data': self.sim_obs[idx],
            'sim_fPrel': self.sim_fPrel[idx],
            'sim_accept_reject': self.sim_accept_reject[idx],
        }