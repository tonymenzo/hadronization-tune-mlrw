import torch
from torch import nn
from torch.utils.data import DataLoader

import numpy as np

class Compressor(nn.Module):
    """
    Autoencoder for compressing event level data in a lower dimensional latent space.
    """
    def __init__(self, input_dim = 100,  output_dim = 100, latent_dim = 20, conditional = True, num_labels = 2):
        """
        Args:
            input_dim (int)    : input dimension of the data
            output_dim (int)   : output dimension of the data
            latent_dim (int)   : dimension of the embedding in the latent space
            conditional (bool) : set to True for passing a condition
            num_labels (int)   : if conditional, set the number of labels for the condition
        """
        super().__init__()
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.latent_dim  = latent_dim
        self.conditional = conditional

        self.init_out_channels_ = 16
        lrelu_slope_            = 0.2
        inter_fc_dim_           = 128
        in_channels             = 1 # pz (in_channels = 5 would correspond to (px, py, pz, E, m))
        kernel_size_conv_       = 3
        kernel_size_pool_       = 2

        # Define the device
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

        # Preset input dimensions for the encoder and decoder
        if self.conditional:
            # Encoder with condition first layer initialization -
            # NN bringing the size of the one hot encoded label (condition) to the
            # same size as the input data
            self.embed_class = nn.Linear(num_labels, self.input_dim)
            self.embed_data  = nn.Conv1d(in_channels, in_channels, kernel_size=1)
            in_channels += 1

            # Decoder with condition first layer initialization -           
            # Adding the number of labels to the size of the latent dim 
            # since they will be merged in the decoding part
            dec_input_size = self.latent_dim + num_labels
        else:
            # For no condition the decoder input will be the same size as the latent latent space
            dec_input_size = self.latent_dim

        
        ####################################################################################################################################################
        # ------------------------------------------------------------------- Encoder -------------------------------------------------------------------- #
        ####################################################################################################################################################

        # Define the encoder layers
        self.features_encode = nn.Sequential(
            nn.Conv1d(in_channels, self.init_out_channels_ * 1, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
                
            nn.Conv1d(self.init_out_channels_ * 1, self.init_out_channels_ * 1, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
            nn.AvgPool1d(kernel_size = kernel_size_pool_, padding = 0),
                
            nn.Conv1d(self.init_out_channels_ * 1, self.init_out_channels_ * 2, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace=True),
                
            nn.Conv1d(self.init_out_channels_ * 2, self.init_out_channels_ * 2, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace=True),
            nn.AvgPool1d(kernel_size = kernel_size_pool_, padding = 0),
                
            nn.Conv1d(self.init_out_channels_ * 2, self.init_out_channels_ * 4, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
                
            nn.Conv1d(self.init_out_channels_ * 4, self.init_out_channels_ * 4, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
            nn.AvgPool1d(kernel_size = kernel_size_pool_, padding = 1)
        )
            
        # Fully connected (fc) layer
        self.fc_encode = nn.Sequential(
            nn.Linear(64 * 10, inter_fc_dim_),
            nn.ReLU(inplace = True),
            nn.Linear(inter_fc_dim_, self.latent_dim)
        )
        
        ####################################################################################################################################################
        # ------------------------------------------------------------------- Decoder -------------------------------------------------------------------- #
        ####################################################################################################################################################

        # Fully connected (fc) layers 
        self.fc_decode = nn.Sequential(
            nn.Linear(dec_input_size, inter_fc_dim_),
            nn.Linear(inter_fc_dim_, 64 * 10),
            nn.ReLU(inplace = True)
        )
        # Define the decoder layers
        self.features_decode = nn.Sequential(
            nn.ConvTranspose1d(self.init_out_channels_* 4 , self.init_out_channels_ * 4, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),

            nn.ConvTranspose1d(self.init_out_channels_ * 4, self.init_out_channels_ * 2, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.init_out_channels_ * 2, self.init_out_channels_ * 2, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
            
            nn.ConvTranspose1d(self.init_out_channels_ * 2, self.init_out_channels_ * 1, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),

            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(self.init_out_channels_ * 1, self.init_out_channels_ * 1, kernel_size = kernel_size_conv_, padding = 1),
            nn.LeakyReLU(lrelu_slope_, inplace = True),
                
            # If input_dim=100 
            nn.ConvTranspose1d(self.init_out_channels_ * 1, in_channels - 1, kernel_size = 38, padding=1),
            nn.LeakyReLU(lrelu_slope_, inplace=True),
            #nn.Conv1d(self.init_out_channels_ * 1, in_channels, kernel_size = kernel_size_conv_, padding=1),
        ) 

    def encode(self, x, c = None):
        """
        Encodes the input and returns the latent dimension.

        Args:
            x : input data
            c : condition vector
        """
        if self.conditional:
            c_hot = c
            # Take the one hot vector and return it in the size of the input data
            embedded_class = self.embed_class(c_hot.float())
            # Reshape for encoder
            embedded_class = embedded_class.view(embedded_class.size()[0], 1, embedded_class.size()[1])
            # Embed the input data
            embedded_input = self.embed_data(x)
            # Merge the input data with the embedded class
            x = torch.cat([embedded_input, embedded_class], dim = 1)
        else:
            x = x.view(-1, 1, x.shape[1])

        # Push through convolutions
        x = self.features_encode(x)
        # Reshape for the fully connected layer
        x = x.view(-1, x.shape[1] * x.shape[2])
        # Push through the fully connected layers
        x = self.fc_encode(x)
        return x
    
    def decode(self, z, c = None):
        """
        Decodes the latent dimension to the output dimension.

        Args:
            z : latent dimension
            c : conditional vector
        """
        if self.conditional:
            # Merge the latent dimenion with the condition vector
            z = torch.cat((z,c), dim = 1)
        # Push through the fully connected layers        
        z = self.fc_decode(z)
        # Reshape for the convolutional layers
        z = z.view(-1, 64, 10)
        # Push through the convolutional layers
        z = self.features_decode(z)
        return z        

    def forward(self, x, c = None):
        """
        Function that defines the network structure.

        Args:
            x : Input data
            c : condition vector
        """

        # Encoding the input
        z = self.encode(x, c)
        # Decoding the latent space
        if self.conditional: dec_z = self.decode(z, c)
        else: dec_z = self.decode(z)
        # Return the decoder output as well as the latent space
        return dec_z , z