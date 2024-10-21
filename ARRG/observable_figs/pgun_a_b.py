from pythia8 import Pythia, Vec4, Event
from math import pi, sqrt, cos, sin

###############################################################################
def sqrtPos(val):
    """
    Returns 0 if val is negative, otherwiswe the sqrt.
    """
    return sqrt(val) if val > 0 else 0.

###############################################################################
class ParticleGun:
    """
    Provides a simple particle gun for hadronic systems using Pythia.

    pythia:  internal Pythia object.
    event:   Pythia event.
    process: Pythia process (the initial setup).
    strings: Pythia event containing the strings, if available.
    pdb:     particle information database.
    rng:     random number generator.
    """
    ###########################################################################
    def __init__(self, a, b, seed = 1, cmds = None):
        """
        Configure and initialize the internal Pythia instance, particle
        data table, and random number generator.

        seed: random number generator seed.
        cmds: optionally pass commands for configuration.
        """
        self.pythia = Pythia("", False)
        if cmds == None: cmds = [
                "StringZ:aLund = %f" % a,
                "StringZ:bLund = %f" % b,
                "1:m0 = 0",
                "2:m0 = 0", 
                "111:m0 = %f" % sqrt(0.135**2 + (1.9*(0.335/sqrt(2.)))**2),
                "211:m0 = %f" % sqrt(0.135**2 + (1.9*(0.335/sqrt(2.)))**2),
                "ProcessLevel:all = off", 
                "HadronLevel:Decay = off",
                "Next:numberShowInfo = 0",
                "Next:numberShowProcess = 0",
                "Next:numberShowEvent = 0",
                "Print:quiet = on",
                "StringFragmentation:TraceColours = on",
                "Random:setSeed = true",
                "Random:seed = %i" % seed,
                "StringFlav:probQQtoQ = 0",
                "StringFlav:probStoUD = 0",
                "StringFlav:mesonUDvector = 0",
                "StringFlav:etaSup = 0", 
                "StringFlav:etaPrimeSup = 0",
                "StringPT:enhancedFraction = 0", 
                #"StringPT:sigma = 0",
                "ParticleData:modeBreitWigner = 0" # Get rid of BW warning message                 
                ]
                

        for cmd in cmds: self.pythia.readString(cmd)
        self.pythia.init()
        self.event   = self.pythia.event
        self.process = self.pythia.process
        self.pdb     = self.pythia.particleData
        self.rng     = self.pythia.rndm
        try: self.strings = self.pythia.strings
        except: pass
    
    ###########################################################################
    def next(self, mode, pe, id1 = 2, id2 = -2,pid = 13):
        """
        Simple method to do the filling of partons into the event record.
        
        mode:  selects the type of event to generate.
               0 = single particle.
               1 = q qbar.
               2 = g g.
               3 = g g g.
               4 = minimal q q q junction topology.
               5 = q q q junction topology with gluons on the strings.
               6 = q q qbar qbar dijunction topology, no gluons.
               7 - 10 = ditto, but with 1 - 4 gluons on string between 
                        junctions.
        pe:    parton energy in GeV.
        pid:   particle ID to generate if mode is 0.
        show:  print the event prior to hadronization.
        """
        # Reset event record to allow for new event.
        self.event.reset()
        try: self.strings.reset()
        except: pass
        
        # Information on a single particle.
        if mode == 0:
            pm = self.pdb.mSel(pid)
            pp = sqrtPos(pe*pe - pm*pm)
            if pe == 0: pe = pm

            # Generate phase-space.
            cThe = 2. * self.rng.flat() - 1.
            sThe = sqrtPos(1. - cThe * cThe)
            phi = 2. * pi * self.rng.flat()

            # Store the particle in the event record.
            iNew = self.event.append( pid, 1, 0, 0, pp * sThe * cos(phi),
                                      pp * sThe * sin(phi), pp * cThe, pe, pm)

            # Generate lifetime, to give decay away from primary vertex.
            self.event[iNew].tau( self.event[iNew].tau0() * self.rng.exp() )
        
        # Information on a q qbar system, to be hadronized.
        elif mode == 1 or mode == 12:
            pid = 2
            pm  = self.pdb.m0(pid)
            pp  = sqrtPos(pe*pe - pm*pm)
            self.event.append(  pid, 23, 101,   0, 0., 0.,  pp, pe, pm)
            self.event.append( -pid, 23,   0, 101, 0., 0., -pp, pe, pm)

        elif mode == 2:
            pid_1 = id1
            pid_2 = id2
            pm_1  = 0.#self.pdb.m0(pid_1)
            pm_2  = 0.#self.pdb.m0(pid_2)
            pe_1 = pe
            pp  = sqrtPos(pe_1*pe_1 - pm_1*pm_1)
            pe_2 = sqrtPos(pp**2 + pm_2**2)
            if pid_1 > 0:
                self.event.append(  pid_1, 23, 101,   0, 0., 0.,  pp, pe_1, pm_1)
                self.event.append(  pid_2, 23,   0, 101, 0., 0., -pp, pe_2, pm_2)
            else:
                self.event.append(  pid_1, 23, 0,   101, 0., 0.,  pp, pe_1, pm_1)
                self.event.append(  pid_2, 23, 101,   0, 0., 0., -pp, pe_2, pm_2)
    
        # Information on a g g system, to be hadronized.
        #elif mode == 2 or mode == 13:
        #    self.event.append( 21, 23, 101, 102, 0., 0.,  pe, pe)
        #    self.event.append( 21, 23, 102, 101, 0., 0., -pe, pe)
    
        # Information on a g g g system, to be hadronized.
        elif mode == 3:
            self.event.append( 21, 23, 101, 102,        0., 0.,        pe, pe)
            self.event.append( 21, 23, 102, 103,  0.8 * pe, 0., -0.6 * pe, pe)
            self.event.append( 21, 23, 103, 101, -0.8 * pe, 0., -0.6 * pe, pe)
    
        elif mode == 4 or mode == 5:
    
            # Need a colour singlet mother parton to define junction origin.
            self.event.append( 1000022, -21, 0, 0, 2, 4, 0, 0,
                          0., 0., 1.01 * pe, 1.01 * pe)
    
            # The three endpoint q q q; the minimal system.
            rt75 = sqrt(0.75)
            self.event.append( 2, 23, 1, 0, 0, 0, 101, 0,
                          0., 0., 1.01 * pe, 1.01 * pe )
            self.event.append( 2, 23, 1, 0, 0, 0, 102, 0,
                          rt75 * pe, 0., -0.5 * pe, pe )
            self.event.append( 1, 23, 1, 0, 0, 0, 103, 0,
                          -rt75 * pe, 0., -0.5 * pe, pe )
    
        # Define the qqq configuration as starting point for adding gluons.
        if mode == 5:
            colNow = [0, 101, 102, 103]
            pQ = [Vec4(), Vec4(0., 0., 1., 0.), Vec4( rt75, 0., -0.5, 0.),
                  Vec4(-rt75, 0., -0.5, 0.)]
    
            # Minimal cos(q-g opening angle), allows more or less nasty events.
            cosThetaMin = 0
    
            # Add a few gluons (almost) at random.
            for nglu in range(0, 5):
                iq, prod = 1 + int( 2.99999 * self.rng.flat() ), float("-inf")
                while prod < cosThetaMin:
                    e =  pe * self.rng.flat()
                    cThe = 2. * self.rng.flat() - 1.
                    phi  = 2. * pi * self.rng.flat()
                    px = e * sqrt(1. - cThe*cThe) * cos(phi)
                    py = e * sqrt(1. - cThe*cThe) * sin(phi)
                    pz = e * cThe
                    prod = (px*pQ[iq].px() + py*pQ[iq].py() + pz*pQ[iq].pz())/e
                colNew = 104 + nglu
                self.event.append( 21, 23, 1, 0, 0, 0, colNew, colNow[iq],
                              px, py, pz, e, 0.)
                colNow[iq] = colNew
                
            # Update daughter range of mother.
            self.event[1].daughters(2, self.event.size() - 1)
    
        # Information on a q q qbar qbar dijunction system, to be hadronized.
        elif mode >= 6:
    
            # The two fictitious beam remnant particles; needed for junctions.
            self.event.append( 2212, -12, 0, 0, 3, 5, 0, 0, 0., 0., pe, pe, 0.)
            self.event.append(-2212, -12, 0, 0, 6, 8, 0, 0, 0., 0., pe, pe, 0.)
    
            # Opening angle between "diquark" legs.
            theta = 0.2
            cThe, sThe = cos(theta), sin(theta)
    
            # Set one colour depending on whether more gluons or not.
            acol = 103 if mode == 6 else 106
    
            # The four endpoint q q qbar qbar; the minimal system.
            # Two additional fictitious partons to make up original beams.
            self.event.append(  2,   23, 1, 0, 0, 0, 101, 0,
                          pe * sThe, 0.,  pe * cThe, pe, 0.)
            self.event.append(  1,   23, 1, 0, 0, 0, 102, 0,
                         -pe * sThe, 0.,  pe * cThe, pe, 0.)
            self.event.append(  2, -21, 1, 0, 9, 0, 103, 0,
                                 0., 0.,  pe       , pe, 0.)
            self.event.append( -2,   23, 2, 0, 0, 0, 0, 104,
                          pe * sThe, 0., -pe * cThe, pe, 0.)
            self.event.append( -1,   23, 2, 0, 0, 0, 0, 105,
                         -pe * sThe, 0., -pe * cThe, pe, 0.)
            self.event.append( -2, -21, 2, 0, 9, 0, 0, acol,
                                 0., 0., -pe       , pe, 0.)
        
            # Add extra gluons on string between junctions.
            if mode == 6:
                self.event.append(1000022,23,5,8,0,0, 0, 0, 0., 0, 0, 0, 0)
            elif mode == 7:
                self.event.append(21,23,5,8,0,0, 103, 106, 0., pe, 0, pe, 0)
            elif mode == 8:
                self.event.append(21,23,5,8,0,0, 103, 108, 0., pe, 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 108, 106, 0.,-pe, 0, pe, 0)
            elif mode == 9:
                self.event.append(21,23,5,8,0,0, 103, 107, 0., pe, 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 107, 108, pe, 0., 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 108, 106, 0.,-pe, 0, pe, 0)
            elif mode == 10:
                self.event.append(21,23,5,8,0,0, 103, 107, 0., pe, 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 107, 108, pe, 0., 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 108, 109, 0.,-pe, 0, pe, 0)
                self.event.append(21,23,5,8,0,0, 109, 106,-pe, 0., 0, pe, 0)

            # Set the extra gluons as daughters from beams.
            for i in [5, 8]: self.event[i].daughter2(self.event.size() - 1)

        # Copy the event to the process.
        self.process = Event(self.event)

        # To have partons shower they must be set maximum allowed scale.
        # Can be set individually to restrict radiation differently.
        if mode == 12 or mode == 13:
            self.event[1].scale(pe);
            self.event[2].scale(pe);
            
            # Now actually do the shower, for range of partons, and max scale.
            # Most restrictive of global and individual applied to each parton.
            self.pythia.forceTimeShower(1, 2, scale);

        # Hadronize the event.
        return self.pythia.next()

    ###########################################################################
    def flatAngle(self, endA, endB, had):
        """
        Sample a flat emission angle.
        
        endA: first end particle.
        endB: second end particle.
        had:  hadronic emission.
        """
        # Boost in the center-of-mass frame.
        com  = endA.p() + endB.p()
        had.bstback(com)

        # Sample the angles.
        cosTheta = 2*self.pythia.rndm.flat() - 1
        sinTheta = sqrtPos(1 - cosTheta**2)
        phi      = 2*pi*self.pythia.rndm.flat()
        pAbs     = had.pAbs()

        # Set and boost back.
        had.p(pAbs * sinTheta * cos(phi), pAbs * sinTheta * sin(phi),
              pAbs * cosTheta, had.e())
        had.bst(com)
