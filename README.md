## Tuning hadronization with Automated Reweighed ReGression (ARRG)

Note that in its current form, the algorithm is only set up to tune two parameters (the a and b Lund parameters) with a single macroscopic observable. Extension to multiple parameters is trivial while an extension to a multi-dimensional macroscopic observable space will require a small amount of work (this is what I will work on next).

### Data-generation
The events were generated with the scripts located in ```event_generator/``` - this directory contains a simplified (no gluon kinks) simulation of a q-qbar system in the center of mass frame hadronizing according to the Lund string model. The Python notebook ```ARRG_event_gen.ipynb``` can be used to generate more events if desired. This particular model only hadronizes to pions - it is simple to add all baryons if desired.

### ARRG algorithm
```lund_weight.py```: PyTorch module for computing the relevant event weights given an alternative parameterization of the Lund fragmentation function. 

```ARRG.py```: The 'heart and soul' of the ARRG algorithm containing a class with functions for loss computation (utilizing ```lund_weight.py``` and a differentiable binning algorithm) and training iteration. Plotting functionality during 'training' can be turned on and off with the ```print_details``` argument.

```train_ARRG.py```: This program prepares the generated data, initializes the dataloaders, initializes tuning hyperparameters, and interfaces with ```ARRG.py``` to perform the parameter tune.

For more details, see the documentation within each script.

### Training data
The training data consists of two datasets:
1. ```exp_observables_dict_a_0.6_b_1.5_sigma_0.335.npy``` contains the macroscopic observable data (multiplicity, sphericity, thrust, ...) points for the 'experimental' or target distributions.
2. ```sim_observables_dict_a_1.5_b_0.6_sigma_0.335.npy``` contains the microscopic accept-reject data in addition to the macroscopic observable data to be reweighed using the algorithm (this file is too big for the git repo, I'll place it on zenodo).