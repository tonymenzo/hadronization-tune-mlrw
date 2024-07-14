import torch
from torch import nn

class LundWeight(nn.Module):
    def __init__(self, params_base, params, over_sample_factor=15.):
        super(LundWeight, self).__init__()
        self.params_base = params_base
        self.params_a = torch.nn.Parameter(params[0].clone(), requires_grad=True)
        self.params_b = torch.nn.Parameter(params[1].clone(), requires_grad=True)
        self.over_sample_factor = over_sample_factor
    
    def zMaxCalc(self, a, b, c):
        # Superfluous constants
        AFROMZERO = 0.02
        AFROMC = 0.01
        
        # Normalization for Lund fragmentation function so that f <= 1.
        # Special cases for a = 0 and a = c.
        aIsZero = (a < AFROMZERO)
        aIsC = (torch.abs(a - c) < AFROMC)
        # Determine position of maximum.
        if aIsZero:
            return b / c if c > b else 1.
        elif aIsC:
            return b / (b + c)
        else:
            zMax = 0.5 * (b + c - torch.sqrt(torch.pow(b - c, 2) + 4 * a * b)) / (c - a)
            if torch.isnan(zMax).any():
                print('zMax_1', zMax)
                print('a', a, 'b', b, 'c', c)
                exit()
            # Grab indicies for special condition
            # Assuming zMax, a, and b are PyTorch tensors of the same shape
            zMax = torch.where((zMax > 0.9999) & (b > 100.), torch.min(zMax, 1. - a / b), zMax)
            return zMax
    
    def likelihood(self, z, mT, a, b, c = torch.tensor(1., requires_grad=True)):
        """
        Compute the likelihood of the Lund fragmentation function
        """
        AFROMZERO = 0.02
        EXPMAX = 10
        
        # Adjust b-parameter
        b_exp = b * torch.pow(mT, 2)
        # Special case for a = 0.
        aIsZero = (a < AFROMZERO)
        # Determine position of maximum.
        zMax = self.zMaxCalc(a, b_exp, c)
        # Be careful of -inf values in aCoeff, very nasty bug to find.
        aCoef = torch.log((1. - z)) - torch.log((1. - zMax))
        if torch.isneginf(aCoef).any():
            print('aCoef is returning -inf value, please check that all z < 1.')
            print('aCoeff', aCoef)

        bCoef = (1. / zMax) - (1. / z)
        cCoef = torch.log(zMax) - torch.log(z)
        fExp = b_exp * bCoef + c * cCoef
        # Special cases for a = 0.
        if not aIsZero:           
            fExp = fExp + a * aCoef
            fVal = torch.exp(torch.clamp(fExp, min=-EXPMAX, max=EXPMAX))
        else:
            fVal = torch.exp(torch.clamp(fExp, min=-EXPMAX, max=EXPMAX))
        return fVal

    def forward(self, z, mT, observable):
        """
        Forward pass of the weight module -- consists of computing the event weights for a given batch
        of training data.
        """
        batch_size = z.shape[0]
        weights = torch.ones(batch_size)
        # Vectorizing the processing as much as possible (still room for improvement)
        for i in range(batch_size):
            # Multiplicity of event in batch
            event_mult = int(observable[i])
            # For multiplte observables the input tensor will have 2 indices (we will 
            # conventionally fill the zeroth position with the event multiplicity of the event)
            #event_mult = int(observable[i, 0])

            # Process accepted values
            accept_indices = z[i, :event_mult, 0] != 0.
            accept_tensor = z[i, :event_mult, 0][accept_indices]
            accept_mT = mT[i, :event_mult][accept_indices]
            accept_weights = self.likelihood(accept_tensor, accept_mT, self.params_a, self.params_b) / self.likelihood(accept_tensor, accept_mT, self.params_base[0], self.params_base[1])
            
            # Process rejected values
            reject_tensor = z[i, :event_mult, 1:]
            reject_weights = torch.ones(event_mult)
            for j in range(event_mult):
                reject_values = reject_tensor[j, reject_tensor[j, :] != 0.]
                if reject_values.numel() > 0:
                    reject_weights_i = (self.over_sample_factor - self.likelihood(reject_values, mT[i, j], self.params_a, self.params_b)) / (self.over_sample_factor - self.likelihood(reject_values, mT[i, j], self.params_base[0], self.params_base[1]))
                    reject_weights[j] = torch.prod(reject_weights_i)
            
            # Compute event weight
            event_weight = torch.prod(accept_weights) * torch.prod(reject_weights)
            weights[i] = event_weight
    
        return weights