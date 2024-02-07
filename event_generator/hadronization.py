"""
# hadronization.py is a part of the MLHAD package.
# Copyright (C) 2023 MLHAD authors (see AUTHORS for details).
# MLHAD is licenced under the GNU GPL v2 or later, see COPYING for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""

from data.PDB import ParticleDatabase
import numpy as np
from pythia8 import Vec4
from data.flavor_selector import prob_flavor_selection
import pgun_a_b_sigma

class Hadronization():
    def __init__(self, a = 0.68, b = 0.98, sigma = 0.335, over_sample_factor = 10.):
        """
        Class contains neccesary functions to produce full hadronization events from 
        trained ML model

        model (string): ---- Path to model file
        conditional (bool) : Conditioned model
        """
        self.pd = ParticleDatabase()
        self.a = a
        self.b = b
        self.sigma = sigma
        self.gen = pgun_a_b_sigma.ParticleGun(a = self.a, b = self.b, sigma = self.sigma)
        self.over_sample_factor = over_sample_factor

    def zMaxCalc(self, a, b, c):
        CFROMUNITY = 0.01
        AFROMZERO = 0.02
        AFROMC = 0.01
        EXPMAX = 50
        # Normalization for Lund fragmentation function so that f <= 1.
        # Special cases for a = 0 and a = c.
        aIsZero = (a < AFROMZERO)
        aIsC = (np.abs(a - c) < AFROMC)
        # Determine position of maximum.
        if aIsZero:
            return b / c if c > b else 1.
        elif aIsC:
            return b / (b + c)
        else:
            zMax = 0.5 * (b + c - np.sqrt((b - c)**2 + 4 * a * b)) / (c - a)
            if zMax > 0.9999 and b > 100.:
                zMax = np.min(zMax, 1. - a / b)
            return zMax
    
    def zFrag(self, a, b, mT, c = 1.):
        """
        Sample longitudinal momentum fraction z with rejected values.
        Translated from Pythia source code - see src/FragmentationFlavZpT.cc
        """
        CFROMUNITY = 0.01
        AFROMZERO = 0.02
        AFROMC = 0.01
        EXPMAX = 50
        OVERSAMPLE_FACTOR = self.over_sample_factor

        # Shape parameters for f(z)
        b = b * mT**2
        
        # Special cases for c = 1, a = 0 and a = c.
        cIsUnity = (abs(c - 1.) < CFROMUNITY)
        aIsZero = (a < AFROMZERO)
        aIsC = (abs(a - c) < AFROMC)
        
        # Determine position of maximum.
        zMax = self.zMaxCalc(a, b, c)
        # Subdivide z range if distribution very peaked near either endpoint.
        peakedNearZero = (zMax < 0.1)
        peakedNearUnity = (zMax > 0.85 and b > 1.)
    
        # Find integral of trial function everywhere bigger than f (dummy start values.).
        fIntLow = fIntHigh = fInt = zDiv = zDivC =  1.0
        accepted_values, rejected_values = [],[]

        if peakedNearZero:
            zDiv = 2.75 * zMax
            fIntLow = zDiv
            if cIsUnity:
                fIntHigh = -zDiv * np.log(zDiv)
            else:
                zDivC = zDiv ** (1. - c)
                fIntHigh = zDiv * (1. - 1./zDivC) / (c - 1.)
            fInt = fIntLow + fIntHigh
    
        elif peakedNearUnity:
            rcb = np.sqrt(4. + (c / b) ** 2)
            zDiv = rcb - 1./zMax - (c / b) * np.log(zMax * 0.5 * (rcb + c / b))
            if not aIsZero:
                zDiv += (a/b) * np.log(1. - zMax)
            zDiv = min(zMax, max(0., zDiv))
            fIntLow = 1. / b
            fIntHigh = 1. - zDiv
            fInt = fIntLow + fIntHigh

        # Choice of z, preweighted for peaks at low or high z. (Dummy start values.)
        z = fPrel = fVal = 0.5
        accept = False
    
        while not accept:
            # Choice of z flat good enough for distribution peaked in the middle;
            # if not this z can be reused as a random number in general.
            z = np.random.rand()
            fPrel = 1.
            # When z_max small use flat below z_div and 1/z^c above z_div.
            if peakedNearZero:
                if fInt * np.random.rand() < fIntLow:
                    z = zDiv * z
                elif cIsUnity:
                    z = zDiv ** z
                    fPrel = zDiv / z
                else:
                    z = (zDivC + (1. - zDivC) * z) ** (1. / (1. - c))
                    fPrel = (zDiv / z) ** c
            # When z_max large use exp( b * (z -z_div) ) below z_div
            # and flat above it.
            elif peakedNearUnity:
                if fInt * np.random.rand() < fIntLow:
                    z = zDiv + np.log(z) / b
                    fPrel = np.exp(b * (z - zDiv))
                else:
                    z = zDiv + (1. - zDiv) * z
            # Evaluate actual f(z) (if in physical range) and correct.
            if 0 < z < 1:
                aCoef = np.log((1. - z) / (1. - zMax))
                bCoef = (1. / zMax - 1. / z)
                cCoef = np.log(zMax / z)
                fExp = b * bCoef + c * cCoef
                if not aIsZero:
                    fExp += a * aCoef
                fVal = np.exp(min(EXPMAX, max(-EXPMAX, fExp)))
                # Probability to accept this choice
                pAccept = fVal / (fPrel * OVERSAMPLE_FACTOR)
                accept = (pAccept > np.random.rand())

                if accept:
                    accepted_values.append(z)
                else:
                    rejected_values.append(z)
            #else:
                #rejected_values.append(z)
        # Conventionally choose the accepted z-value as the first element of the array
        z_accept_reject = np.concatenate((accepted_values, rejected_values), axis=None)
        
        return z, z_accept_reject

    def fz(self, a, b, mT, z, c = 1.):
        """
        Compute likelihood given z.
        """
        b = b * mT**2
        # Special cases for c = 1, a = 0 and a = c.
        cIsUnity = (abs(c - 1.) < CFROMUNITY)
        aIsZero = (a < AFROMZERO)
        aIsC = (abs(a - c) < AFROMC)
        # Determine position of maximum.
        zMax = self.zMaxCalc(a, b, c)
        aCoef = np.log((1. - z) / (1. - zMax))
        bCoef = (1. / zMax - 1. / z)
        cCoef = np.log(zMax / z)
        fExp = b * bCoef + c * cCoef
        if not aIsZero:
            fExp += a * aCoef
        fVal = np.exp(min(EXPMAX, max(-EXPMAX, fExp)))
        return fVal

    def kinematics(self, m_h, E_CM):
        """
        Generates px, py, pz values from trained BNF model to be used in the fragmentation chain
        
        Returns px, py, pz floats
        """
        # Generate Pythia event
        self.gen.next(mode = 1, pe = 100)
        # Sample pT
        pT = self.gen.strings.hads[1].pT()
        # Define transverse mass
        mT = np.sqrt(pT**2 + m_h**2)
        # Sample z
        z, z_accept_reject = self.zFrag(a = self.a, b = self.b, mT = mT)
        # Compute pz
        pz = E_CM * z
        # Sample uniform phi
        phi = 2 * np.pi * np.random.uniform()
        # Return px, py, pz
        return pT * np.cos(phi), pT * np.sin(phi), pz, z_accept_reject, mT

    def fragmentation(self, p1, p2, id1, id2, E_threshold, string_end, E_partons = 50.0):
        """
        This function generates a fragmentation event by boosting to the string systems CM frame, generating
        hadronic emission kinematics, and then boosting back to original frame.
    
        p1 (list): --------- Hadronized end four-momentum components input as list
        p2 (list): --------- Spectator end four-momentum components input as list
        id1 (int): --------- String id corresponding to hadronizing end (p1)
        id2 (int): --------- String id corresponding to non-hadronizing end (p2)
        E_threshold (float): IR center of mass energy cutoff for the fragmentation process
        string_end (string): Fragmenting string end input either 'endA' or 'endB'
        E_partons (float): - Energy of initial parton in the CM frame
    
        Returns new string and hadron ids and momenta
        """
        # Generate new string and hadron IDs
        new_IDs = self.gen.strings.flavor(id1)
        new_string_id = -new_IDs[0]
        new_had_id    = new_IDs[1]
        
        v1 = Vec4(*p1)
        v2 = Vec4(*p2)

        px_i = v1.px()
        py_i = v1.py()
        E_i = v1.e()

        # Boost to the string system COM
        #p_sum = v1 + v2
        #v1.bstback(p_sum)
        
        # Rotate such that v1 is completely along the z-axis
        #theta = v1.theta()
        #phi   = v1.phi()
    
        #v1.rot(0.0, -phi)
        #v1.rot(-theta, phi)
        
        # Sample from new string system CM with energy W^2 = W^+ W^- 
        # Could also just use the energy from the v1 four-vector 
        # E_CM = sqrt((2.*sqrt((p1[0]+px)**2 + (p1[1]+py)**2 + (p1[2])**2)) * (2.*p2[3]))/2.
        E_CM = v1.e()

        # Generate kinematics using the CM energy of string end p1 if above threshold
        if E_CM > E_threshold:
            # Sample px, py, pz
            px, py, pz, z_accept_reject, mT = self.kinematics(self.pd[new_had_id].mass, E_CM)

            # Without boundaries implemented in to the network framework we need to ensure 
            # that we are in a viable kinematic regime i.e. 0.0 < pz < 1.0
            #if pz < 0.0 or pz > 1.0:
                # Continue to sample until we get a viable point
            #    while pz < 0.0 or pz > 1.0:
            #        px, py, pz = self.model_kinematics()
            #pz = pz * E_CM

            # Record the sampled pz and pT  
            #pz_COM = pz / E_CM
            #pT_COM = np.sqrt(px**2 + py**2)
            
            # Create a vector to hold the boosted value of pz
            #w = Vec4(px, py, pz, np.sqrt(self.pd[new_had_id].mass**2 +(px)**2 + (py)**2 + (pz)**2))
            #w.rot(0.0, -phi)
            #w.rot(theta, phi)
            #w.bst(p_sum)
    
            # v1 needs to be in the lab frame
            #v1.rot(0.0, -phi)
            #v1.rot(theta, phi)
            #v1.bst(p_sum)
            # Set pz equal to the boosted back value
            # 
            #pz = w.pz()
    
            # New hadron and string four momenta
            #s = Vec4(-px, -py, v1.pz()-pz, np.sqrt(px**2 + py**2 + (v1.pz()-pz)**2))
            s = Vec4(-px, -py, -pz, np.sqrt(px**2 + py**2 + pz**2))
            h = Vec4(v1.px()+px, v1.py()+py, pz, np.sqrt(self.pd[new_had_id].mass**2 +(v1.px()+px)**2 + (v1.py()+py)**2 + (pz)**2))
            
            # Return numpy arrays
            s = np.array([s.px(), s.py(), s.pz(), s.e()])
            h = np.array([h.px(), h.py(), h.pz(), h.e()])
            
        else: 
            # Else return zero values to signal termination in the fragmentation chain
            new_string_id = 0
            new_had_id = 0
            s = [0,0,0,0]
            h = [0,0,0,0]
            E_CM = 0.0
            z_accept_reject = 0
            mT = 0
        
        #if weighted:
            #return new_string_id, s, new_had_id, h, E_CM, weight, pz_COM, pT_COM
        #else:
        return new_string_id, s, new_had_id, h, E_CM, z_accept_reject, mT

    def fragmentation_chain(self, E_partons, E_threshold = 5.0, endA_id = 2, endB_id = -2, mode = 1, print_details = False):
        """
        Generate a chain of fragmentation events using SWAE kinematics.
    
        E_partons (float): ---- Initial parton energy in GeV (easily modified to accomodate two different parton energies)
        E_threshold (float): -- IR center of mass energy cutoff for the fragmentation process
        endA_pid (int): ------- Initial string flavor for endA
        endB_pid (int): ------- Initial string flavor for endB
        mode (int): ----------- Selects the type of event to generate, mode = 1:  q qbar.
        weighted (bool): ------ Return weights associated with second model of hadronization (assuming conditional model trained on two labels)
        weight_switch (0 or 1): Determines which of the two trained conditions will be used to perform hadronization.
        print_details (bool): - When set to True the details of the entirte fragmentation chain will be output to the console
    
        Returns lists of hadron names, 4-momenta, pids, and (optionally) weights
        """
        # Initialize momentum, pid, name log lists
        endA_p_list, endB_p_list, endC_p_list, had_p_list = [],[],[],[]
        endA_pid_list, endB_pid_list, endC_pid_list, had_pid_list = [],[],[],[]
        endA_name_list, endB_name_list, endC_name_list, had_name_list = [],[],[],[]
        #COM = []
        #if weighted:
        #    weight_list = []
        #    weight = 0
        E_CM_check = [E_partons]

        z_accept_reject_list, mT_list = [],[]
    
        # Initialize kinematics
        if mode == 1:
            endA_m = 0.#pd.pdb.m0(endA_id)
            endB_m = 0.#pd.pdb.m0(endB_id)
            endA_p = np.array([0, 0, np.sqrt(E_partons*E_partons - endA_m*endA_m), E_partons])
            endB_p = np.array([0, 0, -np.sqrt(E_partons*E_partons - endB_m*endB_m), E_partons])
    
            # Record initial system 
            endA_p_list.append(endA_p)
            endB_p_list.append(endB_p)
            endA_pid_list.append(endA_id)
            endB_pid_list.append(endB_id)
            endA_name_list.append(self.pd[endA_id].name)
            endB_name_list.append(self.pd[endB_id].name)
    
            endA_p_0 = [endA_p[0], endA_p[1], endA_p[2], endA_p[3]]
            endB_p_0 = [endB_p[0], endB_p[1], endB_p[2], endB_p[3]]
            
            #sum_had_E = 0
            endA_term, endB_term = 0,0
            term_lim = 2
            had_sum_E_A, had_sum_E_B = 0,0

            pz_COM = 0
            pT_COM = 0
            
            # Generate the chain
            while endA_term < term_lim or endB_term < term_lim:
                endA_p_new = -np.ones(4)
                endB_p_new = np.ones(4)
                # Choose endA (>0.5) or endB (<= 0.5)
                r =  np.random.uniform()
                # If less than 0.5 choose endA
                if r > 0.5:
                    if endA_term < term_lim:
                        # Generate the fragmentation for endA
                        #while endA_p_new[2] < 0:
                        #if weighted:
                        #    endA_id_new, endA_p_new, had_id_new, had_p_new, CM_A_E_check, weight, pz_COM, pT_COM = self.fragmentation(endA_p, endB_p, endA_id, endB_id, E_threshold, 'endA', E_partons, weighted = weighted, weight_switch = weight_switch)
                        #else:
                        endA_id_new, endA_p_new, had_id_new, had_p_new, CM_A_E_check, z_accept_reject, mT = self.fragmentation(endA_p, endB_p, endA_id, endB_id, E_threshold, 'endA', E_partons)
                        # Check if the fragmentation should be logged
                        if endA_p_new[3] > 0 and CM_A_E_check > E_threshold:
                            # endA should always have positive pz
                            #if endA_p_new[2] < 0:
                                #print("String end A should only have positive p_z --- resampling and trying again.")
                            endA_id = endA_id_new 
                            endA_p  = endA_p_new 
                            had_id  = had_id_new 
                            had_p   = had_p_new
                            # Update momenta, pids
                            endA_p_list.append(endA_p.tolist())
                            endB_p_list.append(endB_p.tolist())
                            endC_p_list.append(endA_p.tolist())
                            had_p_list.append(had_p.tolist())
                            endA_pid_list.append(endA_id)
                            endB_pid_list.append(endB_id)
                            endC_pid_list.append(endA_id)
                            had_pid_list.append(had_id)
                            endA_name_list.append(self.pd[endA_id].name)
                            endB_name_list.append(self.pd[endB_id].name)
                            endC_name_list.append(self.pd[endA_id].name)
                            had_name_list.append(self.pd[had_id].name)
                            E_CM_check.append(CM_A_E_check)
                            had_sum_E_A += had_p[3]
                            z_accept_reject_list.append(z_accept_reject)
                            mT_list.append(mT)
                            #if weighted:
                                #weight_list.append(weight)
                            #COM.append(np.array([pz_COM, np.sqrt(pT_COM**2 + self.pd[had_id_new].mass**2)]))
                        else:
                            # Signal endA termination
                            endA_term +=1
                            
                # else choose endB           
                else:
                    if endB_term < term_lim:
                        #while endB_p_new[2] > 0:
                        #if weighted:
                        #    endB_id_new, endB_p_new, had_id_new, had_p_new, CM_B_E_check, weight, pz_COM, pT_COM = self.fragmentation(endB_p, endA_p, endB_id, endA_id, E_threshold, 'endB', E_partons, weighted = weighted, weight_switch = weight_switch)
                        #else:
                        endB_id_new, endB_p_new, had_id_new, had_p_new, CM_B_E_check, z_accept_reject, mT = self.fragmentation(endB_p, endA_p, endB_id, endA_id, E_threshold, 'endB', E_partons)
                        if endB_p_new[3] > 0 and CM_B_E_check > E_threshold:
                            # endB should always have negative pz
                            #if endB_p_new[2] > 0:
                                #print("String end B should only have negative p_z --- resampling and trying again.")

                            endB_id = endB_id_new 
                            endB_p  = endB_p_new 
                            had_id  = had_id_new 
                            had_p   = had_p_new
                            # Update momenta, pids
                            endA_p_list.append(endA_p.tolist())
                            endB_p_list.append(endB_p.tolist())
                            endC_p_list.append(endB_p.tolist())
                            had_p_list.append(had_p.tolist())
                            endA_pid_list.append(endA_id)
                            endB_pid_list.append(endB_id)
                            endC_pid_list.append(endB_id)
                            had_pid_list.append(had_id)
                            endA_name_list.append(self.pd[endA_id].name)
                            endB_name_list.append(self.pd[endB_id].name)
                            endC_name_list.append(self.pd[endB_id].name)
                            had_name_list.append(self.pd[had_id].name)
                            E_CM_check.append(CM_B_E_check)
                            had_sum_E_B += had_p[3]
                            z_accept_reject_list.append(z_accept_reject)
                            mT_list.append(mT)
                            #if weighted:
                                #weight_list.append(weight)
                            #COM.append(np.array([pz_COM, np.sqrt(pT_COM**2 + self.pd[had_id_new].mass**2)]))
                        else:
                            # Signal endB termination
                            endB_term += 1
        # Print the deatils of the entire chain
        if print_details == True:
            # Print the chain results
            print('endA:')
            for i in range(len(endA_p_list)):
                print('Event no: ', i, ', ', 'pid: ', endA_pid_list[i], '(', endA_name_list[i],'), ', 'px: ', '{:.3f}'.format(endA_p_list[i][0]), ' py: ', '{:.3f}'.format(endA_p_list[i][1]), ' pz: ', "{:.3f}".format(endA_p_list[i][2]), ' E: ', '{:.3f}'.format(endA_p_list[i][3]))
       
            print()
            print('endB:')
            for i in range(len(endB_p_list)):
                print('Event no: ', i, ', ', 'pid: ', endB_pid_list[i], '(', endB_name_list[i],'), ', 'px: ', '{:.3f}'.format(endB_p_list[i][0]), ' py: ', '{:.3f}'.format(endB_p_list[i][1]), ' pz: ', "{:.3f}".format(endB_p_list[i][2]), ' E: ', '{:.3f}'.format(endB_p_list[i][3]))
        
            print()
            print('endC:')
            for i in range(len(endC_p_list)):
                print('Event no: ', i+1, ', ', 'pid: ', endC_pid_list[i], '(', endC_name_list[i],'), ', 'px: ', '{:.3f}'.format(endC_p_list[i][0]), ' py: ', '{:.3f}'.format(endC_p_list[i][1]), ' pz: ', "{:.3f}".format(endC_p_list[i][2]), ' E: ', '{:.3f}'.format(endC_p_list[i][3]))
    
            print()
            print('hadron:')
            for i in range(len(had_p_list)):
                print('Event no: ', i+1, ', ', 'pid: ', had_pid_list[i], '(', had_name_list[i],'), ', 'px: ', '{:.3f}'.format(had_p_list[i][0]), ' py: ', '{:.3f}'.format(had_p_list[i][1]), ' pz: ', "{:.3f}".format(had_p_list[i][2]), ' E: ', '{:.3f}'.format(had_p_list[i][3]))

            #if weighted:
            #    print()
            #    print('Event weights:', weight_list)

            print()
            print('COM energies:', E_CM_check)
            # Quick check for energy-momentum conservation 
            #sum_px, sum_py, sum_pz, sum_E = 0,0,0,0
            #for i in range(len(had_p_list)):
            #    sum_px+= had_p_list[i][0]
            #    sum_py+= had_p_list[i][1]
            #    sum_pz+= had_p_list[i][2]
            #    sum_E+= had_p_list[i][3]
            #print()
            #print('hadron sum: ', 'px: ', sum_px, 'py: ', sum_py, 'pz: ', sum_pz, 'E: ', sum_E)
        #if weighted:
        #    return had_name_list, had_p_list, had_pid_list, weight_list, COM
        #else:
            print('Accepted and rejected values of z', z_accept_reject_list)
        return had_name_list, had_p_list, had_pid_list, endC_p_list, z_accept_reject_list, mT_list

if __name__ == "__main__":
    # Generate the hadronization chain for partons with initial energy 100 GeV
    hadronization = Hadronization()
    hadronization.fragmentation_chain(100.0, E_threshold = 5., print_details=True)