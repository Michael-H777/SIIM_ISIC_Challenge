import torch 
from models.building_blocks import * 
from models.base_class import base_model 


class conv_net(base_model): 
    
    def __init__(self, *, in_channels, out_channels, layers=5, channel_growth=16, optimizer=torch.optim.Adam, name=None, **kwargs): 
        base_model.__init__(self)
        self.name = 'Conv_Net' if name is None else name 
        model = torch.nn.Sequential() 
        # make model 
        out_channels = out_channels if isinstance(out_channels, int) else channel_growth*(layers+1)
        model.add_module('Amplify Conv', ConvLayer(in_channels=in_channels, out_channels=channel_growth, kernel_size=1, **kwargs))
        for number in range(layers): 
            in_c, out_c = channel_growth*(number+1), channel_growth*(number+2) if number != layers-1 else out_channels
            model.add_module(f'Conv Layer {number}', ConvLayer(in_channels=in_c, out_channels=out_c, **kwargs))
        # make loss 
        self.model = torch.nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters())
        self.check_attrs()


class d_conv_net(base_model):
    
    def __init__(self, *, in_channels, out_channels, block_layers=5, channel_growth=16, optimizer=torch.optim.Adam, name=None, **kwargs):
        base_model.__init__(self)
        self.name = 'DConv_Net' if name is None else name 
        # make model 
        model = torch.nn.Sequential() 
        model.add_module('Amplify Conv', ConvLayer(in_channels=in_channels, out_channels=channel_growth, **kwargs))
        model.add_module('DenseBlock', DenseBlock(in_channels=channel_growth, out_channels=out_channels, 
                                                  channel_growth=channel_growth, block_layers=block_layers, **kwargs))
        # make loss 
        self.model = torch.nn.DataParallel(model).cuda()
        self.optimizer = optimizer(self.model.parameters())
        self.check_attrs()

