import torch 
from models.base_class import * 
from models.building_blocks import *

class d_unet(torch.nn.Sequential):
    
    def __init__(self, *, in_channels, out_channels, layers=4, channel_growth=16, interpolation='bicubic', 
                 transpose=False,  **kwargs):
        super().__init__()
        
        amplify_factor = 32 
        
        submodule = DenseBlock(in_channels=amplify_factor+channel_growth*layers, out_channels=amplify_factor+channel_growth*layers, 
                               block_layers=6, **kwargs)
        # because we do this from bottom to top, e.g. 
        '''
         amplify (32)                               shrink(32)
        |-- downsampling -----  ----- upsampling --| -> 32 + growth*0
         |-- downsampling ----  ---- upsampling --|  -> 32 + growth*1
          |-- downsampling ---  --- upsampling --|   -> 32 + growth*2
           |-- downsampling --  -- upsampling --|    -> 32 + growth*3
                            bottom                   -> 32 + growth*4
        '''
        # hence (e.g. layers=4), multiple <- {3,2,1,0}
        for multiple in range(layers-1, -1, -1):
            submodule = UNetLayer(channels=amplify_factor+channel_growth*multiple, submodule=submodule, channel_growth=channel_growth, 
                                  interpolation=interpolation, transpose=transpose, **kwargs)
        
        self.add_module('in_layer', DenseBlock(in_channels=in_channels, out_channels=amplify_factor, block_layers=2, kernel_size=1))
        self.add_module('submodule', submodule)
        self.add_module('out_layer', ConvLayer(in_channels=amplify_factor, out_channels=out_channels))

