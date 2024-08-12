import numpy as np
from npy_append_array import NpyAppendArray
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

"""
The following program tuning_data_converter.py converts 
.txt files generated from Pythia into numpy arrays stored 
in a .npy file for use with deep learning frameworks.

Specifically, this program will convert the accepted and 
rejected z values as well has hadron four-momenta data to 
two separate .npy files. These files are ordered according
to when they were generated and must be kept consistent when
utilized during training. 
"""

# Initialize parser
parser = argparse.ArgumentParser()

# Add data_path_accept_reject argument for accept_reject .txt file
parser.add_argument("--data_path_accept_reject", help = "Path to data file containing accepted and rejected z-values.")
# Add data_path_hadrons argument for hadron data .txt file
parser.add_argument("--data_path_hadrons", help = "Path to data file containing hadron info.")
# Add write_path argument for accept-reject data
parser.add_argument("--write_path_accept_reject", help = "Write path for converted accept-reject data.")
# Add write_path argument for hadron data
parser.add_argument("--write_path_hadrons", help = "Write path for converted hadron data.")
# Add print_details argument
parser.add_argument("--print_details", help = "Print progress statements.")

args = parser.parse_args()

if bool(args.print_details) == True:
    print("Input accept-reject .txt file path:", args.data_path_accept_reject)
    print("Input hadron .txt file path:", args.data_path_hadrons)
    print("Writing accept-reject data to:", args.write_path_accept_reject)
    print("Writing hadron data to:", args.write_path_hadrons)

# Read in the lines of the .txt file
with open (args.data_path_accept_reject, 'r') as f:
    lines = f.readlines()

######################################################
#------------- Convert accept-reject ----------------#
######################################################

# Initialize dummy counters
counter = 0

# Initialize padding parameters
npad_accept_reject = 100
npad_event = 50

print("Converting accepted-reject data...")
with tqdm(total = len(lines)) as pbar:
    for line in lines:
        # Events are separated at the new line delimiter
        if line != '\n':
            if counter == 0:
                # Zero-pad the accept-reject array and create array
                arz_i = np.array([np.pad(np.array(line.split(), dtype = float), (0, npad_accept_reject - len(line.split())))])
                counter += 1
            else:
                # Zero-pad the accept-reject array and append
                arz_i = np.append(arz_i, np.array([np.pad(np.array(line.split(), dtype = float), (0, npad_accept_reject - len(line.split())))]), axis = 0)
        else:
            # Zero-pad on the event dimension and create event array
            arz_i = np.pad(arz_i, ((0, npad_event - len(arz_i)),(0,0)))
            # Write to .npy file located at write_path
            with NpyAppendArray(args.write_path_accept_reject) as datafile:
                datafile.append(np.array([arz_i]))
            counter = 0
        pbar.update(1)

# Print the dataset size
pgun_accept_reject = np.load(args.write_path_accept_reject, mmap_mode = 'r')
print("The accept-reject dataset shape is:", pgun_accept_reject.shape)

######################################################
#-------------- Convert hadron data -----------------#
######################################################

print("Converting hadron data...")

with open (args.data_path_hadrons, 'r') as f:
    lines = f.readlines()

# Initialize dummy counters
counter = 0

# Initialize padding parameters
npad_event = 50

with tqdm(total = len(lines)) as pbar:
    for line in lines:
        # Events are separated at the new line delimiter
        if line != '\n':
            if counter == 0:
                # Zero-pad the accept-reject array and create array
                arz_i = np.array([np.array(line.split(), dtype = float)])
                counter += 1
            else:
                # Zero-pad the accept-reject array and append
                arz_i = np.append(arz_i, np.array([np.array(line.split(), dtype = float)]), axis = 0)
        else:
            # Zero-pad on the event dimension and create event array
            arz_i = np.pad(arz_i, ((0, npad_event - len(arz_i)),(0,0)))

            with NpyAppendArray(args.write_path_hadrons) as datafile:
                datafile.append(np.array([arz_i]))
            counter = 0
        pbar.update(1)

# Print the dataset size
pgun_hadrons = np.load(args.write_path_hadrons, mmap_mode = 'r')
print("The hadron dataset shape is:", pgun_hadrons.shape)
print('Finished!')
