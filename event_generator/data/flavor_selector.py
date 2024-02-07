"""
# flavor_selector.py is a part of the MLHAD package.
# Copyright (C) 2022 MLHAD authors (see AUTHORS for details).
# MLHAD is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

import numpy as np
import sys
import pickle
from data.PDB import ParticleDatabase

pdb = ParticleDatabase()
def prob_flavor_selection(str_id):
    """
    Probalistic flavor selector.

    Args:
        str_id (int)      : string end id
        pions_only (bool) : If True, only Pions are allowed as possible hadron outputs

    Returns the new string end and the new hadron output.
    """

    # Load dictionary with the counts of the possible hadron outputs
    pickle_in = open("data/ids_dict.pkl", "rb")
    ids_dict = pickle.load(pickle_in)
    pickle_in.close

    # Computing weights
    pion_ids_list = [-211, 211, 111] # List of all pion ids
    possible_had = [] # Creating a new list of possible output hadrons

    new_ids_dict = {}
    sum_counts = 0

    for i in ids_dict[str_id].keys():
        if i[1] in pion_ids_list:

            possible_had.append(i[1]) # Adding possible hadron outputs
            new_ids_dict[i[1]] = ids_dict[str_id][i] # Adding the counts of the possible hadron outputs in the new_ids_dict
            sum_counts +=ids_dict[str_id][i] # Sum all counts to get the weigths

    weights = [0.5,0.5] # it is the same probability to get pi0 and pi+/-

    # Giving all he possible hadron outputs and the weights, the new hadron will be choosen depending on the weight/probability
    output_had = np.random.choice(possible_had,p=weights)

    # Hadron quark cancellation
    n2 = (abs(output_had) // 100) % 10
    n3 = (abs(output_had) // 10) % 10

    #print("Output Hadron is a Meson")
    # The meson is made of two quarks, where one of them is the input string.
    # The trick we used to get quark concelllation of the hadron doesn't consider the negative sign.
    # so if the input string id is positive, the second quark in the hadron has to be an anti quark (negative id).
    # the new string end will be the anti quark of the second quark of the hadron

    if n2 == abs(str_id):
        if str_id > 0: new_string = n3
        else: new_string = -n3
    elif n3 == abs(str_id):
        if str_id > 0: new_string = n2
        else: new_string = -n2
    else:
        new_string = str_id

    return new_string, output_had

if __name__=="__main__":

    ####################################################
    # Example: probalistic flavor selection (only pions)
    ####################################################
    str_id = -2
    print("input string", str_id, pdb[str_id].name)
    new_string, output_had =  prob_flavor_selection(str_id)
    print("output hadron:", output_had, pdb[output_had].name)
    print("new string id, name: ", new_string, pdb[new_string].name)