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
# Add data_path_fPrel agrument for fPrel.txt file
parser.add_argument("--data_path_fPrel", help = "Path to data file containing fPrel info.")
# Add data_path_mT2 argument for mT2 data .txt file
parser.add_argument("--data_path_mT2", help = "Path to data file containing mT2 info.")
# Add data_path_hadrons argument for hadron data .txt file
parser.add_argument("--data_path_hadrons", help = "Path to data file containing hadron info.")
# Add write_path argument for accept-reject data
parser.add_argument("--write_path_mT2_accept_reject", help = "Write path for converted accept-reject data.")
# Add write_path argument for fPrel data
parser.add_argument("--write_path_fPrel", help = "Write path for converted fPrel data.")
# Add write_path argument for hadron data
parser.add_argument("--write_path_hadrons", help = "Write path for converted hadron data.")
# Add print_details argument
parser.add_argument("--print_details", help = "Print progress statements.")

args = parser.parse_args()

if bool(args.print_details) == True:
    print("Input accept-reject .txt file path:", args.data_path_accept_reject)
    print("Input fPrel .txt file path:", args.data_path_fPrel)
    print("Input hadron .txt file path:", args.data_path_hadrons)
    print("Writing accept-reject data to:", args.write_path_mT2_accept_reject)
    print("Writing fPrel data to:", args.write_path_fPrel)
    print("Writing hadron data to:", args.write_path_hadrons)

# Read in the lines of from mT2 and accept-reject data
with open (args.data_path_accept_reject, 'r') as f:
    lines_ar = f.readlines()

with open(args.data_path_mT2, 'r') as f:
    lines_mT2 = f.readlines()

######################################################
#------------- Convert accept-reject ----------------#
######################################################

# Initialize dummy counters
counter = 0
event_counter = 0

# Initialize padding parameters
npad_accept_reject = 100
npad_event = 105

print("Converting accepted-reject data...")
for i in tqdm(range(len(lines_ar)), ncols = 100):
    # Events are separated at the new line delimiter
    if lines_ar[i] != '\n':
        # Initialize the accept-reject array and make sure it is not the accept-reject delimiter
        if counter == 0 and lines_ar[i] != '&\n':
            # Zero-pad the accept-reject array and create array
            arz_i = np.array([np.pad(np.array(lines_ar[i].split(), dtype = float), (0, npad_accept_reject - len(lines_ar[i].split())))])
            #print(arz_i.shape)
            # Prepend the squared transverse mass
            arz_i = np.insert(arz_i, 0, lines_mT2[i], axis = 1)
            counter += 1
            #print(arz_i.shape)
        elif lines_ar[i] != '&\n':
            # Zero-pad the accept-reject array and append
            arz_I = np.array([np.pad(np.array(lines_ar[i].split(), dtype = float), (0, npad_accept_reject - len(lines_ar[i].split())))])
            # Prepend the squared transverse mass
            arz_I = np.insert(arz_I, 0, lines_mT2[i], axis = 1)
            #print(arz_I)
            #print('arz_I shape:', arz_I.shape)
            arz_i = np.append(arz_i, arz_I, axis = 0)
        else:
            continue
            
    else:
        # Zero-pad on the event dimension and create event array
        arz_i = np.pad(arz_i, ((0, npad_event - len(arz_i)),(0,0)))
        # Write to .npy file located at write_path
        with NpyAppendArray(args.write_path_mT2_accept_reject) as datafile:
            datafile.append(np.array([arz_i]))
        counter = 0
        event_counter += 1
        #if event_counter == n_events: break

# Print the dataset size
pgun_accept_reject = np.load(args.write_path_mT2_accept_reject, mmap_mode = 'r')
print("The accept-reject dataset shape is:", pgun_accept_reject.shape)

######################################################
#-------------- Convert fPrel data ------------------#
######################################################
with open(args.data_path_fPrel, 'r') as f:
    lines_fPrel = f.readlines()

# Initialize dummy counters
counter = 0
event_counter = 0

# Initialize padding parameters
npad_accept_reject = 100
npad_event = 105

for i in tqdm(range(len(lines_fPrel)), ncols = 100):
    # Events are separated at the new line delimiter
    if lines_fPrel[i] != '\n':
        # Initialize the accept-reject array and make sure it is not the accept-reject delimiter
        if counter == 0 and lines_ar[i] != '&\n':
            # Zero-pad the accept-reject array and create array
            arz_i = np.array([np.pad(np.array(lines_fPrel[i].split(), dtype = float), (0, npad_accept_reject - len(lines_fPrel[i].split())))])
            counter += 1
        elif lines_fPrel[i] != '&\n':
            # Zero-pad the accept-reject array and append
            arz_I = np.array([np.pad(np.array(lines_fPrel[i].split(), dtype = float), (0, npad_accept_reject - len(lines_fPrel[i].split())))])
            arz_i = np.append(arz_i, arz_I, axis = 0)
        else:
            continue
            
    else:
        # Zero-pad on the event dimension and create event array
        arz_i = np.pad(arz_i, ((0, npad_event - len(arz_i)),(0,0)))
        # Write to .npy file located at write_path
        with NpyAppendArray(args.write_path_fPrel) as datafile:
            datafile.append(np.array([arz_i]))
        counter = 0
        event_counter += 1
        #if event_counter == n_events: break

pgun_fPrel_monash_prime = np.load(args.write_path_fPrel, mmap_mode = "r")
print("The fPrel dataset shape is:", pgun_fPrel_monash_prime.shape)

######################################################
#-------------- Convert hadron data -----------------#
######################################################

print("Converting hadron data...")

with open (args.data_path_hadrons, 'r') as f:
    lines = f.readlines()

# Initialize dummy counters
counter = 0
event_counter = 0

# Initialize padding parameters
npad_event = 75

# Desired number of events
#n_events = 1e5

with tqdm(total = len(lines), ncols = 100) as pbar:
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
