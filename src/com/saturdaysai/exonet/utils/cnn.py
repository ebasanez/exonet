import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn

class AstroNET_v1(nn.Module):
    def __init__(self, 
                 len_global_lightcurves = 2049, 
                 len_local_lightcurves = 257, 
                 len_extra_parameters = 0,
                 len_fully_connected = 512,
                 input_channels = 1, 
                 output_classes = 2,
                 pooling_type='max'):
        
        super().__init__()
        
        # Pooling dimensional reduction funciton. Order is number of poolings that the
        # column uses. Assumes that:   2*padding = kernel_size-1 and stride = 2
        def pooling_reduction(input_dim, order):
            if order == 1:
                output_dim = int(((input_dim - 1)/2 + 1)//1)
                return output_dim
            else:
                next_dim = int(((input_dim - 1)/2 + 1)//1)
                return pooling_reduction(next_dim, order-1)
        
        
        # General configuration:
        self.input_channels = input_channels
        
        self.len_global_lightcurves = len_global_lightcurves
        self.len_local_lightcurves = len_local_lightcurves
        self.len_extra_parameters = len_extra_parameters
        
        self.len_total_input = (len_global_lightcurves + 
                                len_local_lightcurves + 
                                len_extra_parameters)
        
        self.len_fully_connected = len_fully_connected
        self.output_classes = output_classes
        
                                
        # Calculate the length of the vectors after the convolutional columns
        self.len_global_col = pooling_reduction(self.len_global_lightcurves, 5) * 256
        self.len_local_col  = pooling_reduction(self.len_local_lightcurves, 2) * 32
                                
        # Calculate the input size for the first fully connected layer
        self.len_fc_input = (self.len_global_col + 
                             self.len_local_col + 
                             self.len_extra_parameters * self.input_channels)
        
                                
        # Layers for convolutional columns of the model
        # Layers with same config must be repeated because they will need different weights
        c = self.input_channels
        
        # Convolutions for global view column
        self.conv_5_16_g_a  = nn.Conv1d(  c,  16, kernel_size=5, stride=1, padding=2)
        self.conv_5_16_g_b  = nn.Conv1d( 16,  16, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_32_g_a  = nn.Conv1d( 16,  32, kernel_size=5, stride=1, padding=2)
        self.conv_5_32_g_b  = nn.Conv1d( 32,  32, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_64_g_a  = nn.Conv1d( 32,  64, kernel_size=5, stride=1, padding=2)
        self.conv_5_64_g_b  = nn.Conv1d( 64,  64, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_128_g_a = nn.Conv1d( 64, 128, kernel_size=5, stride=1, padding=2)
        self.conv_5_128_g_b = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_256_g_a = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv_5_256_g_b = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
                                
        # Convolutions for local view column   
        self.conv_5_16_l_a  = nn.Conv1d(  c,  16, kernel_size=5, stride=1, padding=2)
        self.conv_5_16_l_b  = nn.Conv1d( 16,  16, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_32_l_a  = nn.Conv1d( 16,  32, kernel_size=5, stride=1, padding=2)
        self.conv_5_32_l_b  = nn.Conv1d( 32,  32, kernel_size=5, stride=1, padding=2)
        
                                
        # Pooling layers
        # These contain no parameters, so they can be shared
        if pooling_type == 'max':
            self.pool_5_2     = nn.MaxPool1d(5, stride=2, padding=2)
            self.pool_7_2     = nn.MaxPool1d(7, stride=2, padding=3)
        elif pooling_type == 'avg':
            self.pool_5_2     = nn.AvgPool1d(5, stride=2, padding=2)
            self.pool_7_2     = nn.AvgPool1d(7, stride=2, padding=3)
        
        # Dense layers for classification of extracted features
        self.fc_512_a     = nn.Linear(self.len_fc_input, self.len_fully_connected)
        self.fc_512_b     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        self.fc_512_c     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        self.fc_512_d     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        
        self.fc_out       = nn.Linear(self.len_fully_connected, self.output_classes)
        
    def forward(self, xb):  # xb is of size (batch_size, input_channels, len_total_input)
        
        batch_size     = xb.size()[0]
        
        # Extract input of different columns for whole batch and all channels
        gb, lb, eb = torch.split(xb, [self.len_global_lightcurves,
                                      self.len_local_lightcurves,
                                      self.len_extra_parameters],
                                 dim=2)
                            
        # Convolutions for global view
        gb = F.relu(self.conv_5_16_g_a (gb))
        gb = F.relu(self.conv_5_16_g_b (gb))
        gb = self.pool_5_2(gb)        
                                
        gb = F.relu(self.conv_5_32_g_a (gb))
        gb = F.relu(self.conv_5_32_g_b (gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_64_g_a (gb))
        gb = F.relu(self.conv_5_64_g_b (gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_128_g_a(gb))
        gb = F.relu(self.conv_5_128_g_b(gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_256_g_a(gb))
        gb = F.relu(self.conv_5_256_g_b(gb))
        gb = self.pool_5_2(gb)
        
        gb = torch.flatten(gb, 1, 2)  # Flatten channels and features but NOT batches
                                
        # Convolutions for local view
        lb = F.relu(self.conv_5_16_l_a (lb))
        lb = F.relu(self.conv_5_16_l_b (lb))
        lb = self.pool_7_2(lb)        
                                
        lb = F.relu(self.conv_5_32_l_a (lb))
        lb = F.relu(self.conv_5_32_l_b (lb))
        lb = self.pool_7_2(lb)
        
        lb = torch.flatten(lb, 1, 2)
                                
        # Reshape extra features
        eb = torch.flatten(eb, 1, 2)
                                
        # Concatenate results maintaining batch positioning in first dimension
        fb = torch.cat((gb, lb, eb), dim=1)

        # Apply fully connected layers
        fb = F.relu(self.fc_512_a(fb))
        fb = F.relu(self.fc_512_b(fb))
        fb = F.relu(self.fc_512_c(fb))
        fb = F.relu(self.fc_512_d(fb))

        # Output layer
        fb = self.fc_out(fb)
        
        return fb
    

class ExoplaNET_v1(nn.Module):
    def __init__(self, 
                 len_global_lightcurves = 2049, 
                 len_local_lightcurves = 257, 
                 len_secondary_lightcurves = 0, 
                 len_extra_parameters = 0,
                 len_fully_connected = 512,
                 input_channels = 1, 
                 output_classes = 2,
                 pooling_type='max'):
        
        super().__init__()
        
        # Pooling dimensional reduction funciton. Order is number of poolings that the
        # column uses. Assumes that:   2*padding = kernel_size-1 and stride = 2
        def pooling_reduction(input_dim, order):
            if order == 1:
                output_dim = int(((input_dim - 1)/2 + 1)//1)
                return output_dim
            else:
                next_dim = int(((input_dim - 1)/2 + 1)//1)
                return pooling_reduction(next_dim, order-1)
        
        
        # General configuration:
        self.len_global_lightcurves    = len_global_lightcurves
        self.len_local_lightcurves     = len_local_lightcurves
        self.len_secondary_lightcurves = len_secondary_lightcurves
        
        self.len_extra_parameters = len_extra_parameters
        self.input_channels = input_channels
        
        self.len_total_input = (len_global_lightcurves + 
                                len_local_lightcurves + 
                                len_secondary_lightcurves +
                                len_extra_parameters)
        
        self.len_fully_connected = len_fully_connected
        self.output_classes = output_classes
        
                                
        # Calculate the length of the vectors after the convolutional columns
        self.len_global_col     = pooling_reduction(self.len_global_lightcurves, 5) * 256
        self.len_local_col      = pooling_reduction(self.len_local_lightcurves, 2) * 32
        self.len_secondary_col  = pooling_reduction(self.len_secondary_lightcurves, 2) * 32
                                
        # Calculate the input size for the first fully connected layer
        self.len_fc_input = (self.len_global_col + 
                             self.len_local_col + 
                             self.len_secondary_col +
                             self.len_extra_parameters * self.input_channels)
        
                                
        # Layers for convolutional columns of the model
        # Layers with same config must be repeated because they will need different weights
        c = self.input_channels
        
        # Convolutions for global view column
        self.conv_5_16_g_a  = nn.Conv1d(  c,  16, kernel_size=5, stride=1, padding=2)
        self.conv_5_16_g_b  = nn.Conv1d( 16,  16, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_32_g_a  = nn.Conv1d( 16,  32, kernel_size=5, stride=1, padding=2)
        self.conv_5_32_g_b  = nn.Conv1d( 32,  32, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_64_g_a  = nn.Conv1d( 32,  64, kernel_size=5, stride=1, padding=2)
        self.conv_5_64_g_b  = nn.Conv1d( 64,  64, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_128_g_a = nn.Conv1d( 64, 128, kernel_size=5, stride=1, padding=2)
        self.conv_5_128_g_b = nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_256_g_a = nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv_5_256_g_b = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
                                
        # Convolutions for local view column   
        self.conv_5_16_l_a  = nn.Conv1d(  c,  16, kernel_size=5, stride=1, padding=2)
        self.conv_5_16_l_b  = nn.Conv1d( 16,  16, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_32_l_a  = nn.Conv1d( 16,  32, kernel_size=5, stride=1, padding=2)
        self.conv_5_32_l_b  = nn.Conv1d( 32,  32, kernel_size=5, stride=1, padding=2)

        # Convolutions for secondary view column   
        self.conv_5_16_s_a  = nn.Conv1d(  c,  16, kernel_size=5, stride=1, padding=2)
        self.conv_5_16_s_b  = nn.Conv1d( 16,  16, kernel_size=5, stride=1, padding=2)
        
        self.conv_5_32_s_a  = nn.Conv1d( 16,  32, kernel_size=5, stride=1, padding=2)
        self.conv_5_32_s_b  = nn.Conv1d( 32,  32, kernel_size=5, stride=1, padding=2)
                                
        # Pooling layers
        # These contain no parameters, so they can be shared
        if pooling_type == 'max':
            self.pool_5_2     = nn.MaxPool1d(5, stride=2, padding=2)
            self.pool_7_2     = nn.MaxPool1d(7, stride=2, padding=3)
        elif pooling_type == 'avg':
            self.pool_5_2     = nn.AvgPool1d(5, stride=2, padding=2)
            self.pool_7_2     = nn.AvgPool1d(7, stride=2, padding=3)
        
        # Dense layers for classification of extracted features
        self.fc_512_a     = nn.Linear(self.len_fc_input, self.len_fully_connected)
        self.fc_512_b     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        self.fc_512_c     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        self.fc_512_d     = nn.Linear(self.len_fully_connected, self.len_fully_connected)
        
        self.fc_out       = nn.Linear(self.len_fully_connected, self.output_classes)
        
    def forward(self, xb):  # xb is of size (batch_size, input_channels, len_total_input)
        
        batch_size     = xb.size()[0]
        
        # Extract input of different columns for whole batch and all channels
        gb, lb, sb, eb= torch.split(xb, [self.len_global_lightcurves,
                                         self.len_local_lightcurves,
                                         self.len_secondary_lightcurves,
                                         self.len_extra_parameters],
                                    dim=2)
                            
        # Convolutions for global view
        gb = F.relu(self.conv_5_16_g_a (gb))
        gb = F.relu(self.conv_5_16_g_b (gb))
        gb = self.pool_5_2(gb)        
                                
        gb = F.relu(self.conv_5_32_g_a (gb))
        gb = F.relu(self.conv_5_32_g_b (gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_64_g_a (gb))
        gb = F.relu(self.conv_5_64_g_b (gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_128_g_a(gb))
        gb = F.relu(self.conv_5_128_g_b(gb))
        gb = self.pool_5_2(gb)                        
                                
        gb = F.relu(self.conv_5_256_g_a(gb))
        gb = F.relu(self.conv_5_256_g_b(gb))
        gb = self.pool_5_2(gb)
        
        gb = torch.flatten(gb, 1, 2)  # Flatten channels and features but NOT batches
                                
        # Convolutions for local view
        lb = F.relu(self.conv_5_16_l_a (lb))
        lb = F.relu(self.conv_5_16_l_b (lb))
        lb = self.pool_7_2(lb)        
                                
        lb = F.relu(self.conv_5_32_l_a (lb))
        lb = F.relu(self.conv_5_32_l_b (lb))
        lb = self.pool_7_2(lb)
        
        lb = torch.flatten(lb, 1, 2)
        
        # Convolutions for secondary view
        sb = F.relu(self.conv_5_16_s_a (sb))
        sb = F.relu(self.conv_5_16_s_b (sb))
        sb = self.pool_7_2(sb)        
                                
        sb = F.relu(self.conv_5_32_l_a (sb))
        sb = F.relu(self.conv_5_32_l_b (sb))
        sb = self.pool_7_2(sb)
        
        sb = torch.flatten(sb, 1, 2)
                                
        # Reshape extra features
        eb = torch.flatten(eb, 1, 2)
                                
        # Concatenate results maintaining batch positioning in first dimension
        fb = torch.cat((gb, lb, sb, eb), dim=1)

        # Apply fully connected layers
        fb = F.relu(self.fc_512_a(fb))
        fb = F.relu(self.fc_512_b(fb))
        fb = F.relu(self.fc_512_c(fb))
        fb = F.relu(self.fc_512_d(fb))

        # Output layer
        fb = self.fc_out(fb)
        
        return fb