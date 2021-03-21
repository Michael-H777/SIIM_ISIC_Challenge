from packages import * 

from models.building_blocks import * 
from models.conv_net import *
from models.pix2pix import * 
from models.unet import * 

data = torch.ones(64,1,128,128)

model = d_unet(in_channels=1, out_channels=1)

'''
amplify_factor = 32 
channel_growth = 16 
layers =4 
model = DenseBlock(in_channels=amplify_factor+channel_growth*layers, out_channels=amplify_factor+channel_growth*layers, 
                               block_layers=6)
print(model)

'''

print(model.model)
model.set_input(data)
model.forward() 