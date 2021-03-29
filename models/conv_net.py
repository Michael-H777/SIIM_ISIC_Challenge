import torch 
from models.base_class import * 
from models.building_blocks import * 


class conv_net(torch.nn.Sequential): 
    
    def __init__(self, *, in_channels, out_channels, layers=5, channel_growth=16, **kwargs): 
        super().__init__()
        # make model 
        out_channels = out_channels if isinstance(out_channels, int) else channel_growth*(layers+1)
        self.add_module('Amplify Conv', ConvLayer(in_channels=in_channels, out_channels=channel_growth, kernel_size=1, **kwargs))
        for number in range(layers): 
            in_c, out_c = channel_growth*(number+1), channel_growth*(number+2) if number != layers-1 else out_channels
            self.add_module(f'Conv Layer {number}', ConvLayer(in_channels=in_c, out_channels=out_c, **kwargs))


class d_conv_net(torch.nn.Sequential):
    
    def __init__(self, *, in_channels, out_channels, block_layers=5, channel_growth=16, **kwargs):
        super().__init__(self)
        self.add_module('Amplify Conv', ConvLayer(in_channels=in_channels, out_channels=channel_growth, **kwargs))
        self.add_module('DenseBlock', DenseBlock(in_channels=channel_growth, out_channels=out_channels, 
                                                  channel_growth=channel_growth, block_layers=block_layers, **kwargs))

