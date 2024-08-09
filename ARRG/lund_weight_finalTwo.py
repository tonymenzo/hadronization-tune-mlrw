import torch
from torch import nn

class LundWeight(nn.Module):
    def __init__(self, params_base, params, over_sample_factor):
        super(LundWeight, self).__init__()
        self.params_base = params_base
        self.params_a = torch.nn.Parameter(params[0].clone(), requires_grad=True)
        self.params_b = torch.nn.Parameter(params[1].clone(), requires_grad=True)
        self.over_sample_factor = over_sample_factor

        self.AFROMZERO = 0.02
        self.EXPMAX = 50.
        self.AFROMC = 0.01
    
    def zMaxCalc(self, a, b, c):
        # Normalization for Lund fragmentation function so that f <= 1.
        # Special cases for a = 0 and a = c.
        aIsZero = (a < self.AFROMZERO)
        aIsC = (torch.abs(a - c) < self.AFROMC)
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

            # Adjust zMax in numerically unstable regions 
            zMax = torch.where((zMax > 0.9999) & (b > 100.), torch.min(zMax, 1. - a / b), zMax)
            return zMax
    
    def likelihood(self, z, mT, a, b, c = torch.tensor(1., requires_grad=True), z_mask = None, mT_mask = None):
        """
        Compute the likelihood of the Lund fragmentation function

        Args:
            z (torch.Tensor): Input tensor
            mT (torch.Tensor): Transverse mass tensor (can be different shape from z)
            a (torch.Tensor): Parameter a
            b (torch.Tensor): Parameter b
            c (torch.Tensor): Parameter c (default: torch.tensor(1., requires_grad=True))
            z_mask (torch.Tensor, optional): Boolean mask tensor for z (default: None)
            mT_mask (torch.Tensor, optional): Boolean mask tensor for mT (default: None)

        Returns:
            likelihood (torch.Tensor): Computed likelihood values (shape determined by broadcasting rules)
        """

        # Determine the shape after broadcasting
        broadcast_shape = torch.broadcast_shapes(z.shape, mT.shape)

        # If no masks are provided, consider all elements
        if z_mask is None:
            z_mask = torch.ones(z.shape, dtype=torch.bool, device=z.device)
        if mT_mask is None:
            mT_mask = torch.ones(mT.shape, dtype=torch.bool, device=mT.device)

        # Broadcast z, mT, and their masks to the common shape
        z_broad = z.expand(broadcast_shape)
        mT_broad = mT.expand(broadcast_shape)
        z_mask_broad = z_mask.expand(broadcast_shape)
        mT_mask_broad = mT_mask.expand(broadcast_shape)
    
        # Combine masks
        combined_mask = z_mask_broad & mT_mask_broad
        
        # Create a tensor to store the results, initialized with zeros
        likelihood = torch.zeros(broadcast_shape, dtype=z.dtype, device=z.device)
    
        # Only perform calculations on unmasked elements
        z_unmasked = z_broad[combined_mask]
        mT_unmasked = mT_broad[combined_mask]
    
        # Check if we have any unmasked elements to process
        if z_unmasked.numel() > 0:
            # Adjust b-parameter
            b_exp = b * mT_unmasked
            
            # Special case for a = 0.
            aIsZero = (a < self.AFROMZERO)
            
            # Determine position of maximum.
            zMax = self.zMaxCalc(a, b_exp, c)
            
            # Be careful of -inf values in aCoeff, very nasty bug to find.
            aCoef = torch.log(1. - z_unmasked) - torch.log(1. - zMax)
            if torch.isneginf(aCoef).any():
                print('aCoef is returning -inf value, please check that all z < 1.')
                print('aCoeff', aCoef)
    
            bCoef = (1. / zMax) - (1. / z_unmasked)
            cCoef = torch.log(zMax) - torch.log(z_unmasked)
            fExp = b_exp * bCoef + c * cCoef
            
            # Special cases for a = 0.
            if ~aIsZero:
                fExp = fExp + a * aCoef
            
            # Feed through numerical stabilizer
            fVal = torch.exp(torch.clamp(fExp, min=-self.EXPMAX, max=self.EXPMAX))
            
            # Assign computed values back to the likelihood tensor
            likelihood[combined_mask] = fVal
        
        return likelihood

    def forward(self, z, fPrel):
        """
        Forward pass of the weight module -- consists of computing the event weights for a given batch
        of training data.
        """
        batch_size = z.shape[0]
        weights = torch.ones(batch_size)

        # Extract the mT2 values 
        mT2 = z[:, :, 0]
        # Reshape into column tensor
        mT2 = mT2.view(mT2.shape[0], mT2.shape[1], 1)
        # Create a mask for zero values
        mT2_mask = mT2 != 0.

        # Extract the accepted z values
        z_accept = z[:, :, 1]
        # Reshape into column tensor
        z_accept = z_accept.view(z_accept.shape[0], z_accept.shape[1], 1)
        # Remove any zero values
        z_accept_mask = z_accept != 0.

        # Extract the rejected z values
        z_reject = z[:, :, 2:]
        # Reshape into column tensor
        z_reject = z_reject.view(z_reject.shape[0], z_reject.shape[1], z_reject.shape[2])
        # Remove any zero values along the event index
        z_reject_mask = z_reject != 0.

        fPrel_reject = fPrel[:, :, 1:]
        fPrel_reject = fPrel_reject.view(fPrel_reject.shape[0], fPrel_reject.shape[1], fPrel_reject.shape[2])
        fPrel_reject_mask = fPrel_reject != 0.

        # Compute the accept and reject weights
        accept_weights = self.likelihood(z_accept, mT2, self.params_a, self.params_b, z_mask = z_accept_mask, mT_mask = mT2_mask) / self.likelihood(z_accept, mT2, self.params_base[0], self.params_base[1], z_mask = z_accept_mask, mT_mask = mT2_mask)    
        reject_weights = ((self.over_sample_factor * (fPrel_reject * fPrel_reject_mask.masked_fill(z_accept_mask == 0, 1))) - self.likelihood(z_reject, mT2, self.params_a, self.params_b, z_mask = z_reject_mask, mT_mask = mT2_mask)) / ((self.over_sample_factor * (fPrel_reject * fPrel_reject_mask.masked_fill(z_accept_mask == 0, 1))) - self.likelihood(z_reject, mT2, self.params_base[0], self.params_base[1], z_mask = z_reject_mask, mT_mask = mT2_mask))

        # Flatten the weights
        accept_weights = (accept_weights * z_accept_mask).masked_fill(z_accept_mask == 0, 1).prod(dim=2).prod(dim=1)
        reject_weights = (reject_weights * z_reject_mask).masked_fill(z_reject_mask == 0, 1).prod(dim=2).prod(dim=1)
            
        # The final event weight is the product of accepted and rejected weights
        weights = accept_weights * reject_weights
    
        return weights