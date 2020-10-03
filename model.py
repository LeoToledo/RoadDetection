import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

#Fundamental Encoder Module of the network
class fire(nn.Module):
    #size_in is the input_size to the first conv layer
    #squeeze is its output and thus the input of second and third conv layers
    #expand is the output of second and third conv layers
    def __init__(self, size_in, squeeze, expand):
        super(fire, self).__init__()
        
        #Squeeze Phase
        self.conv1 = nn.Conv2d(size_in, squeeze, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(squeeze)
        
        #Expand Phase
        self.conv2 = nn.Conv2d(squeeze, expand, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(expand)
        
        self.conv3 = nn.Conv2d(squeeze, expand, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(expand)
        
        
    def forward(self, x):
        #Squeeze Phase
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        #Expand Phase 1x1 kernel
        out1 = self.conv2(x)
        out1 = self.bn2(out1)
        
        #Expand Phase 3x3 kernel
        out2 = self.conv3(x)
        out2 = self.bn3(out2)
        
        #Concat expands
        x = torch.cat([out1, out2], 1)
        
        return F.relu(x)   
  

#Fire Decoder Module
class fireDec(nn.Module):
    
    def __init__(self, input_size, squeeze_dim, expand_dim):
        
        super(fireDec, self).__init__()
        
        #Expand Layer
        self.expand1x1 = nn.Conv2d(input_size, expand_dim,
                                    kernel_size=1)
        self.expand3x3 = nn.Conv2d(input_size, expand_dim,
                                    kernel_size=3, padding=1)
        
        #Squeeze Layer
        self.squeeze = nn.Conv2d(2*expand_dim, squeeze_dim, 
                                  kernel_size=1)
        
    def forward(self, x):
        
        y = F.relu( self.expand1x1(x) )
        z = F.relu( self.expand3x3(x) )
        x = torch.cat((y, z), 1)
        x = F.relu( self.squeeze(x) )
        return x
      

#Building the network 
class SqueezeNet(nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        
        
        #ENCODE PHASE
        #The number of channels of the input image is 3, cause its a RGB img
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
          
        
        #Fire modules have the architecture (In, Mid, 2*Out)
        self.fire2 = fire(64, 16, 48)
        self.fire3 = fire(96, 16, 48)
        self.fire4 = fire(96, 32, 64)
        self.fire5 = fire(128, 32, 64)
        self.fire6 = fire(128, 48, 64)
        self.fire7 = fire(128, 48, 64)
        self.fire8 = fire(128, 48, 64)
        self.fire9 = fire(128, 64, 64)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0, stride=2)
        
        #DECODE PHASE
        #Firedec modules have the architecture (In, out, 2*mid)
        self.firedec10 = fireDec(64, 48, 32)
        self.upsample10 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.firedec11 = fireDec(48, 64, 32)
        self.upsample11 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.firedec12 = fireDec(64, 32, 48)
        #self.upsample12 = nn.Upsample(scale_factor=2, mode='nearest')
                
        self.fire_deconv13 = nn.ConvTranspose2d(32, 24, kernel_size=6, padding=2, stride=2)
        self.fire_deconv14 = nn.ConvTranspose2d(24, 1, kernel_size=6, padding=2, stride=2)
        
    def forward(self, x):
        
        #Encode Phase
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x , indice1= F.max_pool2d(x, (2,2), stride=2, return_indices=True)
        
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.fire4(x)
        
        x, indice2= F.max_pool2d(x, (2,2), stride=2, return_indices=True)
        
        x = self.fire5(x)
        x = self.fire6(x)
        x = self.fire7(x)
        x = self.fire8(x)
        
        x= F.max_pool2d(x, (2,2), stride=1) 
        
        x = self.fire9(x)
        
        x = self.conv2(x)
        
        #Decode Phase
        x = self.firedec10(x)
        x = self.upsample10(x)
        
        x = self.firedec11(x)
        x = self.upsample11(x)
        
        x = self.firedec12(x)
        #x = self.upsample12(x)
        
        x = self.fire_deconv13(x)
        x = F.relu(x)
        
        x = self.fire_deconv14(x)
    
        return F.sigmoid(x)
    
net = SqueezeNet()
# print(net)