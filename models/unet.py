import torch 
from models.building_blocks import *
from models.base_class import base_model 

class d_unet(torch.nn.Module):
    
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
        
        self.model = torch.nn.Sequential(
            DenseBlock(in_channels=in_channels, out_channels=amplify_factor, block_layers=2, kernel_size=1), 
            submodule, 
            ConvLayer(in_channels=amplify_factor, out_channels=out_channels)
        )

    def forward(self, input_data): 
        return self.model(input_data)


class d_unet_cuda(base_model):
    
    def __init__(self, loss=['l1'], optimizer=torch.optim.Adam, name=None, **kwargs):
        
        super().__init__(loss)
        self.name = 'Dense_UNet' if name is None else name 
        
        model = d_unet(**kwargs)
        
        self.model = torch.nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters())
        self.check_attrs()
        
    def forward(self): 
        self.prediction = self.input_data - self.model.forward(self.input_data)
        return self.prediction