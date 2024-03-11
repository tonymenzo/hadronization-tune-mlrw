import numpy as np
import scipy.stats as st
import sys
import pythia8
from pythia8 import Pythia, Vec4, Event, StringZ, StringEnd, ColSinglet, HadronLevel
from math import pi, sqrt, cos, sin

def bin_assigner(bins,val):

    # This function simply outputs the bin assignment of a given value val for a given choice of binning bins

    number_of_bins = len(bins)
    bin_number=0
    if val >= bins[-1]:
        bin_number = number_of_bins - 1
    if val < bins[0]:
        bin_number = 0
    else:
        for bin_number_aux in range(number_of_bins-1):
            if bins[bin_number_aux]<= val < bins[bin_number_aux+1]:
                bin_number = bin_number_aux
    return bin_number

def sqrtPos(val):
    """
    Returns 0 if val is negative, otherwiswe the sqrt.
    """
    return sqrt(val) if val > 0 else 0.

def pTgauss(pT,sigma):

    return pT*(np.sqrt(2*np.pi)/sigma)*st.norm(loc=0.0,scale=sigma).pdf(pT)

def cdf_pythia(pT,sigma):
    # to turn the pythia pt diff into a uniformly distributed number between 0 and 1
    return 1.0-np.exp(-0.5*(pT/sigma)**2)

def pythia_function_pT(pT,param):
    if pT < param[0] or pT >= param[0]+param[1]:
        return 0.0
    else:
        return 1.0/param[1]#pT*(np.sqrt(2*np.pi)/sigma)*st.norm(loc=0.0,scale=sigma).pdf(pT)

def trial_function_pT(pT,param):
    mu = param[0]
    sigma = param[1]
#    if pT < param[0]-param[1] or pT >= param[0]+param[1]:
#        return 0.0
#    else:
#        return 1.0/(2*param[1])
    return pT*(np.sqrt(2*np.pi)/sigma)*st.norm(loc=mu,scale=sigma).pdf(pT)
    
def log_pythia_function_pT(pT,param):
    if pT < param[0] or pT >= param[0]+param[1]:
        return -np.inf
    else:
        return -np.log(param[1])#pT*(np.sqrt(2*np.pi)/sigma)*st.norm(loc=0.0,scale=sigma).pdf(pT)

def log_trial_function_pT(pT,param):
    mu = param[0]
    sigma = param[1]
#    if pT < param[0]-param[1] or pT >= param[0]+param[1]:
#        return 0.0
#    else:
#        return 1.0/(2*param[1])
    return np.log(pT)+np.log(np.sqrt(2*np.pi)/sigma)+st.norm(loc=mu,scale=sigma).logpdf(pT)

def lund(a,b,sample):
        ### this function mimicks Pythia's zLund. It takes as input two parameters, a and b, and a given sample [mT^2,z] and inputs the Lund function value normalized such that f(zMax) = 1

        z = sample[1]
        beff = b*sample[0]

        ### I initialize two ints to asses whether zMax is too low or too high
        peakednearzero = 0
        peakednearunity = 0
        
        ### zMax determination
        if a == 0.0:
            if 1.0 > beff:
                zMax = beff
            else:
                zMax = 1.0
        elif a == 1.0:
            zMax = beff/(beff+1.0)
        else:
            zMax = 0.5 * (beff + 1.0 - np.sqrt( (beff - 1.0)**2 + 4* a * beff)) / (1.0 - a)
        if (zMax > 0.9999 and beff > 100):
             zMax = np.min([zMax, 1.0 - a/beff])

    # get current regime
        if zMax < 0.1:
            peakednearzero = 1
        if zMax > 0.85 and beff > 1.0:
            peakednearunity = 1

        # right now I'll ignore whether it is too small or too large but I'll print it just to keep in mind that it does happen

    # if zMax too small
        if peakednearzero == 1:
            # print("Near Zero")
            pass
            zDiv = 2.75 * zMax
            fIntLow = zDiv
            fIntHigh = -zDiv * np.log(zDiv)
            fInt = fIntLow + fIntHigh
            fZ = 1.0 if z < zDiv else zDiv/z
            #if z > 0.0 and z < 1.0:
            #    z = zDiv * z if z < zDiv else zDiv/z
        #return 0.0

    # if zMax too large 
        elif peakednearunity == 1:
            # print("Near Unity")
        #return 0.0
            pass
            rcb = np.sqrt(4. + (1.0 / beff)**2)
            zDiv = rcb - 1.0/zMax - (1.0 / beff) *np.log( zMax * 0.5 * (rcb + 1.0 / beff) )+(a/beff) * np.log(1. - zMax)
            zDiv = np.min([zMax, np.max([0., zDiv])])
            fIntLow = 1.0 / beff
            fIntHigh = 1. - zDiv
            fInt = fIntLow + fIntHigh
            fZ = np.exp( beff * (z - zDiv) ) if z < zDiv else 1.0
            #if z > 0.0 and z < 1.0:
            #    z =  zDiv + np.log(z) / beff if z < zDiv else zDiv + (1. - zDiv) * z

        # Lund function itself 
        if 1==1:
            if z > 0.0 and z<1.0:
                #fExp = beff * (0.0 - 1. / z)+ 1.0 * np.log(1.0 / z) + a * np.log( (1. -z) )
                fExp = beff * (1. / zMax - 1. / z)+ 1.0*np.log(zMax / z) + a * np.log( (1. - z) / (1. - zMax) )
                fZ = np.exp( np.max( [-50, np.min( [50, fExp]) ]) ) # 50 is chosen to avoid too large exponents
            else:
                fZ = 0.0
                
        return fZ
        
def log_lund(a,b,sample):
        ### this function mimicks Pythia's zLund. It takes as input two parameters, a and b, and a given sample [mT^2,z] and inputs the Lund function value normalized such that f(zMax) = 1

        z = sample[1]
        beff = b*sample[0]

        ### I initialize two ints to asses whether zMax is too low or too high
        peakednearzero = 0
        peakednearunity = 0
        
        ### zMax determination
        if a == 0.0:
            if 1.0 > beff:
                zMax = beff
            else:
                zMax = 1.0
        elif a == 1.0:
            zMax = beff/(beff+1.0)
        else:
            zMax = 0.5 * (beff + 1.0 - np.sqrt( (beff - 1.0)**2 + 4* a * beff)) / (1.0 - a)
        if (zMax > 0.9999 and beff > 100):
             zMax = np.min([zMax, 1.0 - a/beff])

    # get current regime
        if zMax < 0.1:
            peakednearzero = 1
        if zMax > 0.85 and beff > 1.0:
            peakednearunity = 1

        # right now I'll ignore whether it is too small or too large but I'll print it just to keep in mind that it does happen

    # if zMax too small
        if peakednearzero == 1:
            # print("Near Zero")
            pass
            zDiv = 2.75 * zMax
            fIntLow = zDiv
            fIntHigh = -zDiv * np.log(zDiv)
            fInt = fIntLow + fIntHigh
            fZ = 1.0 if z < zDiv else zDiv/z
            #if z > 0.0 and z < 1.0:
            #    z = zDiv * z if z < zDiv else zDiv/z
        #return 0.0

    # if zMax too large 
        elif peakednearunity == 1:
            # print("Near Unity")
        #return 0.0
            pass
            rcb = np.sqrt(4. + (1.0 / beff)**2)
            zDiv = rcb - 1.0/zMax - (1.0 / beff) *np.log( zMax * 0.5 * (rcb + 1.0 / beff) )+(a/beff) * np.log(1. - zMax)
            zDiv = np.min([zMax, np.max([0., zDiv])])
            fIntLow = 1.0 / beff
            fIntHigh = 1. - zDiv
            fInt = fIntLow + fIntHigh
            fZ = np.exp( beff * (z - zDiv) ) if z < zDiv else 1.0
            #if z > 0.0 and z < 1.0:
            #    z =  zDiv + np.log(z) / beff if z < zDiv else zDiv + (1. - zDiv) * z

        # Lund function itself 
        if 1==1:
            if z > 0.0 and z<1.0:
                #fExp = beff * (0.0 - 1. / z)+ 1.0 * np.log(1.0 / z) + a * np.log( (1. -z) )
                fExp = beff * (1. / zMax - 1. / z)+ 1.0*np.log(zMax / z) + a * np.log( (1. - z) / (1. - zMax) )
                fExp = np.max( [-50, np.min( [50, fExp]) ])
            else:
                fExp = 0.0
                
        return fExp

def lund_unnorm(a,b,sample):
        ### this function mimicks Pythia's zLund. It takes as input two parameters, a and b, and a given sample [mT^2,z] and inputs the Lund function value normalized such that f(zMax) = 1

        z = sample[1]
        beff = b*sample[0]

        ### I initialize two ints to asses whether zMax is too low or too high
        peakednearzero = 0
        peakednearunity = 0
        
        ### zMax determination
        if a == 0.0:
            if 1.0 > beff:
                zMax = beff
            else:
                zMax = 1.0
        elif a == 1.0:
            zMax = beff/(beff+1.0)
        else:
            zMax = 0.5 * (beff + 1.0 - np.sqrt( (beff - 1.0)**2 + 4* a * beff)) / (1.0 - a)
        if (zMax > 0.9999 and beff > 100):
             zMax = np.min([zMax, 1.0 - a/beff])

    # get current regime
        if zMax < 0.1:
            peakednearzero = 1
        if zMax > 0.85 and beff > 1.0:
            peakednearunity = 1

        # right now I'll ignore whether it is too small or too large but I'll print it just to keep in mind that it does happen

    # if zMax too small
        if peakednearzero == 1:
            # print("Near Zero")
            pass
            zDiv = 2.75 * zMax
            fIntLow = zDiv
            fIntHigh = -zDiv * np.log(zDiv)
            fInt = fIntLow + fIntHigh
            fZ = 1.0 if z < zDiv else zDiv/z
            #if z > 0.0 and z < 1.0:
            #    z = zDiv * z if z < zDiv else zDiv/z
        #return 0.0

    # if zMax too large 
        elif peakednearunity == 1:
            # print("Near Unity")
        #return 0.0
            pass
            rcb = np.sqrt(4. + (1.0 / beff)**2)
            zDiv = rcb - 1.0/zMax - (1.0 / beff) *np.log( zMax * 0.5 * (rcb + 1.0 / beff) )+(a/beff) * np.log(1. - zMax)
            zDiv = np.min([zMax, np.max([0., zDiv])])
            fIntLow = 1.0 / beff
            fIntHigh = 1. - zDiv
            fInt = fIntLow + fIntHigh
            fZ = np.exp( beff * (z - zDiv) ) if z < zDiv else 1.0
            #if z > 0.0 and z < 1.0:
            #    z =  zDiv + np.log(z) / beff if z < zDiv else zDiv + (1. - zDiv) * z

        # Lund function itself 
        if 1==1:
            if z > 0.0 and z<1.0:
                #fExp = beff * (0.0 - 1. / z)+ 1.0 * np.log(1.0 / z) + a * np.log( (1. -z) )
                fExp = beff * (- 1. / z)+ 1.0*np.log(1.0 / z) + a * np.log( (1. - z))
                fZ = np.exp( np.max( [-50, np.min( [50, fExp]) ]) ) # 50 is chosen to avoid too large exponents
            else:
                fZ = 0.0
                
        return fZ
        
def log_lund(a,b,sample):
        ### this function mimicks Pythia's zLund. It takes as input two parameters, a and b, and a given sample [mT^2,z] and inputs the Lund function value normalized such that f(zMax) = 1

        z = sample[1]
        beff = b*sample[0]

        ### I initialize two ints to asses whether zMax is too low or too high
        peakednearzero = 0
        peakednearunity = 0
        
        ### zMax determination
        if a == 0.0:
            if 1.0 > beff:
                zMax = beff
            else:
                zMax = 1.0
        elif a == 1.0:
            zMax = beff/(beff+1.0)
        else:
            zMax = 0.5 * (beff + 1.0 - np.sqrt( (beff - 1.0)**2 + 4* a * beff)) / (1.0 - a)
        if (zMax > 0.9999 and beff > 100):
             zMax = np.min([zMax, 1.0 - a/beff])

    # get current regime
        if zMax < 0.1:
            peakednearzero = 1
        if zMax > 0.85 and beff > 1.0:
            peakednearunity = 1

        # right now I'll ignore whether it is too small or too large but I'll print it just to keep in mind that it does happen

    # if zMax too small
        if peakednearzero == 1:
            # print("Near Zero")
            pass
            zDiv = 2.75 * zMax
            fIntLow = zDiv
            fIntHigh = -zDiv * np.log(zDiv)
            fInt = fIntLow + fIntHigh
            fZ = 1.0 if z < zDiv else zDiv/z
            #if z > 0.0 and z < 1.0:
            #    z = zDiv * z if z < zDiv else zDiv/z
        #return 0.0

    # if zMax too large 
        elif peakednearunity == 1:
            # print("Near Unity")
        #return 0.0
            pass
            rcb = np.sqrt(4. + (1.0 / beff)**2)
            zDiv = rcb - 1.0/zMax - (1.0 / beff) *np.log( zMax * 0.5 * (rcb + 1.0 / beff) )+(a/beff) * np.log(1. - zMax)
            zDiv = np.min([zMax, np.max([0., zDiv])])
            fIntLow = 1.0 / beff
            fIntHigh = 1. - zDiv
            fInt = fIntLow + fIntHigh
            fZ = np.exp( beff * (z - zDiv) ) if z < zDiv else 1.0
            #if z > 0.0 and z < 1.0:
            #    z =  zDiv + np.log(z) / beff if z < zDiv else zDiv + (1. - zDiv) * z

        # Lund function itself 
        if 1==1:
            if z > 0.0 and z<1.0:
                #fExp = beff * (0.0 - 1. / z)+ 1.0 * np.log(1.0 / z) + a * np.log( (1. -z) )
                fExp = beff * (1. / zMax - 1. / z)+ 1.0*np.log(zMax / z) + a * np.log( (1. - z) / (1. - zMax) )
                fExp = np.max( [-50, np.min( [50, fExp]) ])
            else:
                fExp = 0.0
                
        return fExp

        
def pythia_function_z(param,sample):
        loc = param[0]
        scale = param[1]
        z = sample[1]
        if z < loc or z >= loc+scale:
            return -np.inf
        else:
            return 1.0/scale#lund(a,b,sample)

def trial_function_z(param,sample):
        '''
        a = param[0]
        b = param[1]
        z = sample[1]
        beff = b*sample[0]
        return lund(a,b,sample)
        '''
        a = param[0]
        b = param[1]
        beff = b*sample[0]
        return st.beta(a=np.max([a,1.0]),b=np.max([beff,1.0])).pdf(sample[1])
        
def log_pythia_function_z(param,sample):
        loc = param[0]
        scale = param[1]
        z = sample[1]
        if z < loc or z >= loc+scale:
            return -np.inf
        else:
            return -np.log(scale)#lund(a,b,sample)


def log_trial_function_z(param,sample):
        a = param[0]
        b = param[1]
        z = sample[1]
        beff = b*sample[0]
        zvals = np.linspace(0,1,1000)
        return log_lund(a,b,sample)-np.log(np.sum((zvals[1]-zvals[0])*np.array(list(map(lambda zval: lund(a,b,[sample[0],zval]),zvals)))))
        '''

        if sample[0] >= mT2vals[0] and sample[0] < mT2vals[-1]:
            ind_mass = bin_assigner(mT2vals,sample[0])
            return log_lund(a,b,sample)-np.log(lund_norm_vals[ind_mass])
        else:
            #zvals = np.linspace(0,1,1000)
            return log_lund(a,b,sample)-np.log(np.sum((zvals[1]-zvals[0])*np.array(list(map(lambda zval: lund(a,b,[sample[0],zval]),zvals)))))
        
        a = param[0]
        b = param[1]
        beff = b*sample[0]
        return st.beta(a=np.max([a,1.0]),b=np.max([beff,1.0])).logpdf(sample[1])
        '''
        
def trial_function_full(param,sample):
        pT, mT2, z = sample
        mu, sigma, a, b = param
        beff = b*mT2
        zvals = np.linspace(0,1,1000)
        return pT*(np.sqrt(2*np.pi)/sigma)*st.norm(loc=mu,scale=sigma).pdf(pT) * st.beta(a=np.max([a,1.0]),b=np.max([beff,1.0])).pdf(sample[1])
        

def log_pythia_function_full(param,sample):
        pT, mT2, z = sample
        #'''
        mu, sigma, loc_z, scale_z = param
        return np.log(pT)+np.log(np.sqrt(2*np.pi)/sigma)+st.norm(loc=0.0,scale=sigma).logpdf(pT)+st.uniform(loc=loc_z,scale=scale_z).logpdf(z)
        '''
        loc_pT, scale_pT, loc_z, scale_z = param
        return st.uniform(loc=loc_pT,scale=scale_pT).logpdf(pT)+st.uniform(loc=loc_z,scale=scale_z).logpdf(z)
        '''


def log_trial_function_full(param,sample):
        pT, mT2, z = sample
        mu, sigma, a, b = param
        aeff, beff = a*mT2, b*mT2
        
        zvals = np.linspace(0,1,1000)
        return np.log(pT)+np.log(np.sqrt(2*np.pi)/sigma)+st.norm(loc=0.0,scale=sigma).logpdf(pT) + log_lund(a,b,[mT2,z])-np.log(zvals[1]-zvals[0])-sp.logsumexp(np.array(list(map(lambda zval: log_lund(a,b,[mT2,zval]),zvals))))
            #np.sum((zvals[1]-zvals[0])*np.array(list(map(lambda zval: lund(a,b,[sample[0],zval]),zvals)))))


# Write own derived UserHooks class.

class MyUserHooks(pythia8.UserHooks):

    # Constructor creates anti-kT jet finder with (-1, R, pTmin, etaMax).
    def __init__(self,sigmaQ, aLund, bLund,aExtraSQuark,aExtraDiquark,rFactC,rFactB):
        pythia8.UserHooks.__init__(self)
        self.counter = 0
        self.abs_counter = 0
        self.supremum = 1.01#self.supremum_pT * self.supremum_z
        self.sigmaQ = sigmaQ
        self.aLund = aLund
        self.bLund = bLund
        self.aExtraSQuark = aExtraSQuark
        self.aExtraDiquark = aExtraDiquark
        self.rFactC = rFactC
        self.rFactB = rFactB
        self.splits = []


    # Allow to change Fragmentation parameters...
    def canVetoProcessLevel(self): 
        return True

    # This changes the parameters of the Pythia models
    def doVetoProcessLevel(self, process):
        last_particle_id = np.abs(process.back().id())
        #print(last_particle_id)
        if last_particle_id >= 3:
            return True
        return False
        
    # Allow to change Fragmentation parameters...
    def canChangeFragPar(self): 
        return True

    # This changes the parameters of the Pythia models
    def doChangeFragPar(self, flavPtr, zPtr, pTPtr, idEnd, m2Had,iParton, sEnd):
        return True

    # This vetos a given hadron on a given string end
    def doVetoFragmentation(self, *args):#had, sEnd):
        # check whether we are in a final hadronization or not
        if(len(args)==2):# and self.counter < 1):
            # we have Hadron, StringEnd
            # print(args[1].GammaOld,args[1].GammaNew,args[1].xPosOld, args[1].xPosNew, args[1].xPosHad, args[1].xNegOld, args[1].xNegNew,args[1].xNegHad)
            #if np.abs(args[1].flavNew.id)>=3:
            #    return True
            zHad = args[1].zHad
            mT2Had = args[1].mT2Had
            pTHad = np.sqrt((args[1].pxHad-args[1].pxOld)**2+(args[1].pyHad-args[1].pyOld)**2)
            self.abs_counter += 1
            #print(pTHad,np.sqrt(args[1].pxHad**2+args[1].pyHad**2),np.sqrt(args[0].px()**2+args[0].py()**2))
            if np.allclose(args[1].mHad**2+args[1].pxHad**2+args[1].pyHad**2,mT2Had) == False:
                print("Wrong mT2 definition")
                print(args[1].mHad**2+args[1].pxHad**2+args[1].pyHad**2,mT2Had)
            #print(zHad,pTHad,mT2Had,args[1].flavOld.id,args[1].flavNew.id,args[1].idHad,args[0].pz(),args[0].pT())
            #print(args[1].pxHad, args[1].pyHad, args[0].pz(), args[1].pxOld, args[1].pyOld)
            self.splits.append([args[1].pxHad, args[1].pyHad, args[0].pz(), args[0].e()])
            #self.splits.append([zHad,args[1].pxHad,args[1].pyHad,args[1].mHad,args[1].flavNew.id,args[1].idHad,args[1].pxOld,args[1].pyOld,args[1].flavOld.id,args[0].status()])
            #self.splits.append([args[0].status()])
            #[zHad,pTHad,mT2Had,args[1].flavOld.id,args[1].flavNew.id,args[1].idHad])
            '''
            log_w0 = log_pythia_function_full([0.0,self.sigmaQ,0.0,1.0],[pTHad,mT2Had,zHad])#log_pythia_function_pT(pTHad,[0.0,2.0])+log_pythia_function_z([0.0,1.0],[mT2Had,zHad])
            if zHad > 0.0 and zHad < 1:
                log_w1 = log_trial_function_full([0.0,self.sigmaQ,self.aLund,self.bLund],[pTHad,mT2Had,zHad])
                #log_trial_function_pT(pTHad,[0.0,sigma_new])+log_trial_function_z([aLund, bLund],[mT2Had,zHad])
            else:
                return True

            random_coin = np.random.rand()
            
            # Here is the vetoing itself
            
            if np.log(random_coin)+np.log(self.supremum) > log_w1-log_w0:#np.log(w1)-np.log(w0):
                self.supremum = np.max([self.supremum,np.exp(log_w1-log_w0)])#w1/w0])
                return True
            '''

            self.counter+=1
            if args[1].mHad <= 0.14:#1==1:#np.abs(args[0].id())==211:
                # uncomment the counter if I only want first hadronization
                #self.supremum = np.max([self.supremum,np.exp(log_w1-log_w0)])#w1/w0])
                #print(str(args[1].mHad)+'\t'+str(pTHad)+'\t'+str(args[1].mT2Had)+'\t'+str(args[1].zHad)+'\t'+str(self.counter))
                pass
            pass
        else:
            # we have had1, had2, s1, s2
            #if np.abs(args[2]))>=3 or np.abs(args[3])>=3:
            #    return True
            pass
        # Do not veto events that got this far.

        return False




###############################################################################
class Reweighted_Hadronization:
    """
    Provides a simple Reweighted Hadronization chains for hadronic systems using Pythia.

    pythia:  internal Pythia object.
    event:   Pythia event.
    process: Pythia process (the initial setup).
    strings: Pythia event containing the strings, if available.
    pdb:     particle information database.
    rng:     random number generator.
    """
    ###########################################################################
    def __init__(self, sigmaQ=0.335/np.sqrt(2), aLund = 0.68, bLund = 0.98,seed = 1, cmds = None):
        """
        Configure and initialize the internal Pythia instance, particle
        data table, and random number generator.

        seed: random number generator seed.
        cmds: optionally pass commands for configuration.
        """
        self.pythia = Pythia("", False)
        if cmds == None: cmds = []
        cfg = ["ProcessLevel:all = off", "HadronLevel:Decay = off",
               "Next:numberShowInfo = 0", "Next:numberShowProcess = 0",
               "Next:numberShowEvent = 0", "Print:quiet = off",
               "StringFragmentation:TraceColours = on","Fragmentation:setVertices = off", "Check:event = false",
                "Random:setSeed = true",  "Random:seed = %i" % seed ,"111:mayDecay = false",
                "StringFlav:probStoUD=0.","StringFlav:mesonUDvector=0.","StringFlav:probQQtoQ = 0.0","StringFlav:etaSup=0.0","StringFlav:etaPrimeSup=0.0",
                "StringPT:sigma="+str(sigmaQ*np.sqrt(2)),"StringZ:aLund = "+str(aLund),"StringZ:bLund = "+str(bLund),
                "StringZ:aExtraSQuark = 0.0","StringZ:aExtraDiquark = 0.0","StringZ:rFactC = 0.0", "StringZ:rFactB = 0.0",
                "StringZ:usezUniform = off","StringPT:usepTUniform = off","StringPT:maxpT = 1.0"]
        for cmd in cfg + cmds: self.pythia.readString(cmd)
        self.sigmaQ = sigmaQ
        self.aLund = aLund
        self.bLund = bLund
        self.myUserHooks = MyUserHooks(sigmaQ, aLund, bLund,0,0,0,0)
        self.pythia.setUserHooksPtr(self.myUserHooks)
        self.pythia.init()
        self.event   = self.pythia.event
        self.process = self.pythia.process
        self.pdb     = self.pythia.particleData
        self.rng     = self.pythia.rndm
        try: self.strings = self.pythia.strings
        except: pass
    
    ###########################################################################
    def next(self):
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
        mod:   optional function mod(event) which can modify the event.
        """
        # Reset event record to allow for new event.
        self.event.reset()
        
        try: self.strings.reset()
        except: pass
        try: self.myUserHooks.counter = 0
        except: pass
        try: self.myUserHooks.abs_counter = 0
        except: pass
        try: self.myUserHooks.splits = []
        except: pass

        # ID of q
        pid = 2

        # energy of each quark
        pe = 20.0

        # mass of each quark
        pm  =0.0* self.pythia.particleData.m0(pid)

        # resulting pz
        pp  = sqrtPos(pe*pe - pm*pm)

    # add particles to event record

        self.pythia.event.append(  pid, 23, 101,   0, 0., 0.,  -pp, pe, pm)
        self.pythia.event.append( -pid, 23,   0, 101, 0., 0., pp, pe, pm)
        # Copy the event to the process.
        self.pythia.process = Event(self.pythia.event)

        # Hadronize the event.
        self.pythia.next()
        #return self.pythia.next()
        
class Reweighted_Hadronization_eeZ:
    """
    Provides a simple Reweighted Hadronization chains for hadronic systems using Pythia.

    pythia:  internal Pythia object.
    event:   Pythia event.
    process: Pythia process (the initial setup).
    strings: Pythia event containing the strings, if available.
    pdb:     particle information database.
    rng:     random number generator.
    """
    ###########################################################################
    def __init__(self, nevents=1, aLund = 0.68, bLund = 0.98,sigmaQ = 0.335/np.sqrt(2),aExtraSQuark=0,aExtraDiquark=0.97,rFactC=1.32,rFactB=0.855,seed = 1, cmds = None):
        """
        Configure and initialize the internal Pythia instance, particle
        data table, and random number generator.

        seed: random number generator seed.
        cmds: optionally pass commands for configuration.
        """
        self.pythia = Pythia("", False)
        if cmds == None: cmds = []
        cfg = ["ProcessLevel:all = on", "HadronLevel:Decay = on","Tune:ee = 7",
        	"Main:numberOfEvents = "+str(nevents), "Beams:idA = 11", "Beams:idB = -11", "Beams:eCM = 91.2","WeakSingleBoson:ffbar2gmZ = on",
               "Next:numberShowInfo = 0", "Next:numberShowProcess = 0",
               "Next:numberShowEvent = 0", "Print:quiet = on",
               "StringFragmentation:TraceColours = off","Fragmentation:setVertices = off", "Check:event = false",
                "Random:setSeed = true",  "Random:seed = %i" % seed ,
                "StringPT:sigma="+str(sigmaQ*np.sqrt(2)),"StringZ:aLund = "+str(aLund),"StringZ:bLund = "+str(bLund),
                "111:mayDecay = false","211:mayDecay = false","StringFlav:probStoUD=0.","StringFlav:mesonUDvector=0.","StringFlav:probQQtoQ = 0.0","StringFlav:etaSup=0.0","StringFlav:etaPrimeSup=0.0",
                "StringPT:sigma="+str(sigmaQ*np.sqrt(2)),"StringZ:aLund = "+str(aLund),"StringZ:bLund = "+str(bLund),
                "StringZ:aExtraSQuark = "+str(aExtraSQuark),"StringZ:aExtraDiquark = "+str(aExtraDiquark),"StringZ:rFactC = "+str(rFactC), "StringZ:rFactB = "+str(rFactB),
                #"StringZ:usezUniform = off","StringPT:usepTUniform = off","StringPT:maxpT = 1.0"
]
        for cmd in cfg + cmds: self.pythia.readString(cmd)
        self.sigmaQ = sigmaQ
        self.aLund = aLund
        self.bLund = bLund
        self.aExtraSQuark = aExtraSQuark
        self.aExtraDiquark = aExtraDiquark
        self.rFactC = rFactC
        self.rFactB = rFactB
        self.myUserHooks = MyUserHooks(sigmaQ, aLund, bLund,aExtraSQuark,aExtraDiquark,rFactC,rFactB)
        self.pythia.setUserHooksPtr(self.myUserHooks)
        self.pythia.init()
        self.event   = self.pythia.event
        self.process = self.pythia.process
        self.pdb     = self.pythia.particleData
        self.rng     = self.pythia.rndm
        try: self.strings = self.pythia.strings
        except: pass
    
    ###########################################################################
    def next(self):
        """
        Simple method to do the filling of partons into the event record.
        
        """
        # Reset event record to allow for new event.
        #self.event.reset()
        
        #try: self.strings.reset()
        #except: pass
        try: self.myUserHooks.counter = 0
        except: pass
        try: self.myUserHooks.abs_counter = 0
        except: pass
        try: self.myUserHooks.splits = []
        except: pass
        
        # Hadronize the event.
        self.pythia.next()
        #return self.pythia.next()
'''
nevents = int(1e1)
hf = Reweighted_Hadronization_eeZ(nevents=nevents) 
nevent = 0
for nevent in range(nevents):
    print(nevent)
    print(hf.myUserHooks.splits)
    hf.next()
    for nprt, prt in enumerate(hf.pythia.event):
        if prt.isFinal() and prt.isVisible():
            #print(nevent,nprt)
            print(prt.pz(),prt.pT(),prt.id())
    #print(hf.pythia.event.list())
'''
