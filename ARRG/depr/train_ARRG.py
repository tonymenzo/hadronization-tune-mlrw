from ARRG import *
import yaml

class ObservableDataset(Dataset):
	"""
	Converts observable dataset into PyTorch syntax.
	"""
	def __init__(self, data):
		self.data = data

	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		sample = self.data[idx]
		return sample

# Load experimental observables
exp_observable_dict   = np.load('exp_observables_dict_a_0.6_b_1.5_sigma_0.335.npy', allow_pickle = True).item()
# Create an 'experimental' observable array
exp_observable = np.column_stack((exp_observable_dict['multiplicity'], exp_observable_dict['sphericity']))

# Load simulated event data
sim_events_dict = np.load(r'/home/tonym/Code/MAGIC_data/sim_events_dict_a_1.5_b_0.6_sigma_0.335.npy', allow_pickle = True).item()
# Extract relevant kinematic information
sim_kinematics_z  = sim_events_dict['z_accept_reject']
sim_kinematics_mT = sim_events_dict['mT']

# Extract observables from simulated events dict
sim_observable_dict = {key: sim_events_dict.get(key) for key in ['multiplicity', 'sphericity']}
sim_observable = np.column_stack((sim_observable_dict['multiplicity'], sim_observable_dict['sphericity']))

# Print dataset shapes
print('Experimental observable shape:', exp_observable.shape)
print('Simulated observable shape:', sim_observable.shape)
print('Simulated z shape:', sim_kinematics_z.shape)
print('Simulated mT shape: ', sim_kinematics_mT.shape)

# Convert into torch objects
sim_observable = torch.Tensor(sim_observable)
sim_kinematics_z = torch.Tensor(sim_kinematics_z)
sim_kinematics_mT = torch.Tensor(sim_kinematics_mT)
exp_observable   = torch.Tensor(exp_observable)

# Prepare data for DataLoader
sim_observable = ObservableDataset(sim_observable)
sim_kinematics_z = ObservableDataset(sim_kinematics_z)
sim_kinematics_mT = ObservableDataset(sim_kinematics_mT)
exp_observable   = ObservableDataset(exp_observable)

# Set batch size -- TBD: Implement batch size scheduler
batch_size = 5000 

# Initialize data-loaders
sim_observable_dataloader = DataLoader(sim_observable, batch_size = batch_size, shuffle = False)
sim_kinematics_z_dataloader = DataLoader(sim_kinematics_z, batch_size = batch_size, shuffle = False)
sim_kinematics_mT_dataloader = DataLoader(sim_kinematics_mT, batch_size = batch_size, shuffle = False)
exp_observable_dataloader = DataLoader(exp_observable, batch_size = batch_size, shuffle = False)

# Training hyperparameters
epochs = 2
over_sample_factor = 15.0
learning_rate = 0.1
# Length of event buffer
dim_multiplicity = sim_kinematics_z_dataloader.dataset.data.shape[1]
dim_accept_reject = sim_kinematics_z_dataloader.dataset.data.shape[2]

print('Each event has been zero-padded to a length of', dim_multiplicity)
print('Each emission has been zero-padded to a length of', dim_accept_reject)

# Define base parameters of simulated data (a, b)
params_base = torch.tensor([1.5, 0.6])
# If params_init is set equal to None, the tuned parameters are initialized to the base parameters
params_init = None
#params_init = torch.tensor([0.6, 1.5])

print_details = True
results_dir = r'./ARRG_a_b_tune'

# Create a training instance
macroscopic_trainer = ARRG(epochs = epochs, dim_multiplicity = dim_multiplicity, dim_accept_reject = dim_accept_reject, over_sample_factor = over_sample_factor,
								   params_base = params_base, sim_observable_dataloader = sim_observable_dataloader, sim_kinematics_z_dataloader = sim_kinematics_z_dataloader, 
								   sim_kinematics_mT_dataloader = sim_kinematics_mT_dataloader, exp_observable_dataloader = exp_observable_dataloader, print_details = print_details, 
								   results_dir = results_dir, params_init = params_init)

# Set the optimizer
optimizer = torch.optim.Adam(macroscopic_trainer.weight_nexus.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(macroscopic_trainer.weight_nexus.parameters(), lr=learning_rate)

# Set learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode= 'min', factor = 0.2, patience = 3)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.1)
#scheduler = None

# Train!
a_b_final, a_b_search = macroscopic_trainer.train_ARRG(optimizer = optimizer, scheduler = scheduler)
	 
print("Training complete!")
print("Final parameters: a =", a_b_final[0], 'b =', a_b_final[1])

"""
# Write out a model summary as well as the search space to a yaml file
model_dict = {'batch_size': batch_size, 'epochs': epochs, 'over_sample_factor': over_sample_factor, 'learning_rate_init': learning_rate, 
			  'params_base': params_base.detach().numpy(), 'params_final': a_b_final, 'params_search': a_b_search}

PATH_model_summary = results_dir + '/ARRG_a_b_tune_summary.yaml'
with open(PATH_model_summary, 'w') as f:
	yaml.dump(model_dict, f)
print('Model summary saved to', PATH_model_summary)
"""