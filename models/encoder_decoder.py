import torch 
from models.base_class import * 
from models.building_blocks import *


class encoder(torch.nn.Module):
    
    def __init__(self, *, in_channels, layers, channel_growth, **kwargs):
        super().__init__()
        self.layers = ['Ampify 0']
        self.skip_layer = str(layers-1)
        setattr(self, 'Ampify 0', ConvLayer2D(in_channels=in_channels, out_channels=channel_growth, kernel_size=1))
        self.channels = []
        for layer_num in range(1, layers+1): 
            block = DenseBlock(in_channels=layer_num*channel_growth, 
                               out_channels=(layer_num+1)*channel_growth, **kwargs)
            pool = Conv2dDown(channels=(layer_num+1)*channel_growth)
            setattr(self, f'Layer {layer_num} Dense', block)
            setattr(self, f'Layer {layer_num} Pool', pool)
            self.layers.extend([f'Layer {layer_num} Dense', f'Layer {layer_num} Pool'])
            if layer_num in [layers-1, layers]:
                self.channels.append((layer_num+1)*channel_growth)
        
    def forward(self, input_data): 
        skip = None
        
        for index, layer_name in enumerate(self.layers): 
            layer = getattr(self, layer_name)
            input_data = layer.forward(input_data)
            if layer_name == f'Layer {self.skip_layer} Pool':
                skip = input_data
                
        return skip, input_data
    
    
class decoder(torch.nn.Module):
    
    def __init__(self, *, out_channels, layers, channel_growth, **kwargs):
        super().__init__()
        channel_configs = [[layer_num, (layer_num+1) * channel_growth, layer_num*channel_growth] for layer_num in range(layers,-1,-1)]
        channel_configs[1][1] += channel_configs[0][2]
        channel_configs[-1][-1] = out_channels
        self.layers = []
        
        for layer_num, in_channels, out_channels in channel_configs:
            block = DenseBlock(in_channels=in_channels, out_channels=out_channels, **kwargs)
            up = torch.nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            setattr(self, f'Layer {layer_num} Dense', block)
            current_layer = [f'Layer {layer_num} Dense']
            if layer_num != 0:
                setattr(self, f'Layer {layer_num} Up', up)
                current_layer += [f'Layer {layer_num} Up']
            self.layers.extend(current_layer)
        
        self.skip_layer = channel_configs[1][0]
        
    def forward(self, input_data): 
        skip, bottom = input_data
        
        for layer_name in self.layers:
            layer = getattr(self, layer_name)
            
            if layer_name == f'Layer {self.skip_layer} Dense':
                bottom = layer(torch.cat([bottom, skip], 1))
            else:
                bottom = layer(bottom)
        return bottom 
        
        
class FC_classifier(torch.nn.Sequential): 
    
    def __init__(self, *, input_size, output_size, FC_layers):
        super().__init__()
        
        print(f'{self.__class__.__name__} {input_size=}')
        fc_configs = [[0, input_size, input_size//128], 
                      [1, input_size//128, input_size//128], 
                      ['Exit', input_size//128, output_size]]
        
        for layer_num, in_size, out_size in fc_configs:
            self.add_module(f'Layer {layer_num}', torch.nn.Linear(in_size, out_size))
            #self.add_module(f'Layer {layer_num} dropout', torch.nn.Dropout(0.2)) if layer_num != 'Exit' else None
            self.add_module(f'Layer {layer_num} Activation', torch.nn.LeakyReLU(0.2)) if layer_num != 'Exit' else None


class Patch_classifier(torch.nn.Module): 
    
    def __init__(self, *, skip_channels, bottom_channels, bottom_size, **kwargs):
        super().__init__() 
        
        layers = bottom_size // 2 
        
        self.skip_module = torch.nn.Sequential(Conv2dUp(channels=skip_channels))
        
        self.classify_module = torch.nn.Sequential()

        channels = skip_channels+bottom_channels
        for index in range(layers):            
            up = Conv2dDown(channels=channels)
            dense = DenseBlock(in_channels=channels, out_channels=channels)
            
            self.classify_module.add_module(f'Layer {index} Up', up)
            self.classify_module.add_module(f'Layer {index} DenseBlock', dense)
            channels //= 2 
        
    def forward(self, input_data): 
        skip, bottom = input_data
        
        skip = self.skip_module(skip)
        bottom = torch.cat([bottom, skip], 1)
        
        for layer in self.classify_module:
            bottom = layer(bottom)
            
        return bottom
    