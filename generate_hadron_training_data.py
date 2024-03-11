import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.stats as st
from hadronization_function import *
import pythia8
import time
from tqdm import tqdm
from pythia8 import Pythia, Vec4, Event, Sphericity,StringZ, StringEnd, ColSinglet, HadronLevel, Thrust

sph = Sphericity(powerIn=1) # powerIn = 1 to get linear version for C and D parameter, powerIn = 2 to get sphericity
thr = Thrust()

seed = 45426752362536458828

def event_info(Event, hadrons_dim):
    """
    Function returning hadron-level event info i.e. hadron four-momenta
    """
    list_of_info = np.zeros((hadrons_dim, 4))
    tag = 0.0
    nparticle = 0
    nparticle_min = 11
    nparticle_max = 100

    # Collect macroscopic event level info
    
    # Feed in event to Sphericity instance
    sph.analyze(Event)
    #print('Sphericity:', sph.sphericity())
    #print('Aplanarity:', sph.aplanarity())
    # Feed in event to Thrust instance
    thr.analyze(Event)
    #print('Thurst:', thr.thrust())
    #print('tMajor:', thr.tMajor())
    #print('tMinor:', thr.tMinor())
    #print('Oblateness: ', thr.oblateness()) 

    # Collect microscopic event level info (particle cloud)
    for particle in Event:
        if np.abs(particle.status()) == 23 and particle.id() == 5:
            tag = 1.0
        if particle.isFinal():# and prt.isCharged():
            #nCharged+=1.0
            if particle.isVisible() and (particle.status() == 83.0 or particle.status() == 84.0):
                # Record the hadron kinematics              
                list_of_info[int(nparticle)] = [particle.px(), particle.py(), particle.pz(), particle.e()]
                nparticle += 1.0
        # Don't allow fragmentation chains larger then nparticle_max
        if nparticle >= nparticle_max:
            break
    # Don't allow fragmentation chains shorter than nparticle_min
    if nparticle < nparticle_min:
        return None, None
    else:
        #print('Event info: ', list_of_info)
        return list_of_info, tag

# Total number of events to generate
nevents = 1e6
# Path to save 
results_dir = '../'

start = time.time()

# Params are aLund, bLund, sigmaQ, aExtraSQuark, aExtraDiquark, rFactC, rFactB

# "Simulation" parameter values (default Monash)
sim_params = np.array([0.6, 1.5, 0.335/np.sqrt(2), 0., 0.97, 1.32, 0.855])

# "Experimental" parameter values
#exp_params = np.array([0.9465845, 1.3782454, 0.2103422, 0., 0.97, 1.32, 0.855])

sim_run = Reweighted_Hadronization_eeZ(nevents = nevents, aLund = sim_params[0], bLund = sim_params[1], sigmaQ = sim_params[2], aExtraSQuark = sim_params[3], aExtraDiquark = sim_params[4], rFactC = sim_params[5], rFactB = sim_params[6])
#exp_run = Reweighted_Hadronization_eeZ(nevents = nevents, aLund = exp_params[0], bLund = exp_params[1], sigmaQ = exp_params[2], aExtraSQuark = exp_params[3], aExtraDiquark = exp_params[4], rFactC = exp_params[5], rFactB = exp_params[6])

sim_tag = np.zeros(int(nevents))
#exp_tag = np.zeros(int(nevents))

# Number of allowed hadrons (upper limit)
hadrons_dim = 100
# Number of 'observables' to collect (we want the four-momentum of each hadron)
individual_hadron_dim = 4
sim_hadrons = np.zeros((int(nevents), int(hadrons_dim), individual_hadron_dim))
#exp_hadrons = np.zeros((int(nevents), int(hadrons_dim), individual_hadron_dim))

# Number of allowed parton-level splittings
splits_dim = 100
# Number of 'observables' to collect (we want zHad, px, py, m, flavNew, idHad, pxOld, pyOld, flavOld, status)
# Collected observables [px, py, pz, E]
individual_split_dim = 4
sim_splits = np.zeros((int(nevents), int(splits_dim), individual_split_dim))
#exp_splits = np.zeros((int(nevents), int(splits_dim), individual_split_dim))

# Generate 'simulated' dataset
nevent = 0
pbar = tqdm(total = nevents, ncols = 100)
while nevent < nevents: #in tqdm(range(int(nevents)), ncols = 100):
    # Generate e+e- event
    sim_run.next()
    # Collect split and hadron info
    splits_aux = sim_run.myUserHooks.splits
    #print('splits_aux', splits_aux)
    if len(splits_aux) > 0 and len(splits_aux) <= splits_dim:
        sim_splits[nevent,:len(splits_aux)] = np.array(splits_aux)
        info_per_event, tag_per_event = event_info(sim_run.pythia.event, hadrons_dim)
        #print('info_per_event', info_per_event)
        if tag_per_event != None:
            sim_hadrons[nevent] = info_per_event
            #sim_tag[nevent] = tag_per_event
            nevent += 1
            pbar.update(1)

pbar.close()
print('Total generated events: ', nevent)
#sim_splits  = sim_splits[:nevent]
sim_hadrons = sim_hadrons[:nevent]
#sim_tag     = sim_tag[:nevent]
print(sim_hadrons.shape)
#print(sim_hadrons)

# Save data
#np.save(results_dir + 'sim_splits_jet_1e5_6.npy',  sim_splits)
np.save(results_dir + 'exp_hadrons_a_0.6_b_1.5_nevents_1e6.npy', sim_hadrons)
"""
# Generate 'experimental' dataset
nevent = 0
pbar = tqdm(total = nevents, ncols = 100)
while nevent < nevents: #in tqdm(range(int(nevents)), ncols = 100):
    # Generate e+e- event
    exp_run.next()
    # Collect split info
    splits_aux = exp_run.myUserHooks.splits
    if len(splits_aux) > 0 and len(splits_aux) <= splits_dim:
        exp_splits[nevent,:len(splits_aux)] = np.array(splits_aux)
        info_per_event, tag_per_event = event_info(exp_run.pythia.event, hadrons_dim)
        if tag_per_event != None:
            exp_hadrons[nevent] = info_per_event
            exp_tag[nevent] = tag_per_event
            nevent += 1
            pbar.update(1)

pbar.close()
print('Total generated events: ', nevent)
exp_splits  = exp_splits[:nevent]
exp_hadrons = exp_hadrons[:nevent]
exp_tag     = exp_tag[:nevent]
"""

# Save data
#np.save(results_dir + rf'exp_splits_jet_1e5_6.npy',  exp_splits)
#np.save(results_dir + rf'exp_hadrons_jet_1e5_6.npy', exp_hadrons)

print('It took', time.time()-start, 'seconds.')