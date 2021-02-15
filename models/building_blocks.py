import torch
import numpy as np 


def make_activation(function_type):
    if function_type == 'relu': 
        return torch.nn.ReLU(inplace=True)
    elif function_type == 'leaky_relu':
        return torch.nn.LeakyReLU(inplace=True)
    elif function_type == 'elu':
        return torch.nn.ELU(inplace=True)
    elif function_type == 'gelu':
        return torch.nn.GELU()
    else:
        raise NotImplementedError(f'activation type {function_type} undefined in code')


def init_layers(layer):
    # https://github.com/SaoYan/DnCNN-PyTorch/blob/master/utils.py
    
    layer_name = layer.__class__.__name__ 
    if layer_name.find('Conv') != -1: 
        torch.nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
    elif layer_name.find('BatchNorm') != -1: 
        # what are these numbers and where they came from?????????
        layer.weight.data.normal_(mean=0, std=np.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        torch.nn.init.constant_(layer.bias.data, 0.0)
    return None


class ChannelAttention(torch.nn.Module): 
    
    # https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf
    # with revision 
    
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__() 
        self.Attention_Pool = torch.nn.AdaptiveAvgPool2d(1)
        self.Attention_Conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=in_channels, 
                                              kernel_size=1, padding=0)
        self.Attention_Activation = torch.nn.Sigmoid()
        self.init_layers()
        
    def init_layers(self): 
        torch.nn.init.kaiming_normal_(self.Attention_Conv.weight.data, mode='fan_in')
        
    def forward(self, input_data):
        attention = self.Attention_Pool(input_data)
        attention = self.Attention_Conv(attention)
        attention = self.Attention_Activation(attention)
        return input_data * attention


class ConvLayer(torch.nn.Sequential): 
    
    def __init__(self, *, in_channels, out_channels, attention=False, kernel_size=5, 
                 dilation=1, dropout=False, use_bn=True, activation='gelu'):
        super(ConvLayer, self).__init__()
        
        # calculate padding after dilation 
        dilated_space = (kernel_size // 2) * (dilation-1)
        dilated_kernel = kernel_size + dilated_space*2
        padding = (dilated_kernel - 1) // 2
        # Conv is must have; but BN, dropout, activation are dependent on model architecture, 
        # which is controlled by changing input param when buildind said model 
        self.add_module('Conv', torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                kernel_size=kernel_size, stride=1, 
                                                dilation=dilation, padding=padding, padding_mode='reflect'))
        self.add_module('BatchNorm', torch.nn.BatchNorm2d(num_features=out_channels)) if use_bn else None
        self.add_module('Dropout', torch.nn.Dropout(dropout, inplace=True)) if isinstance(dropout, float) else None 
        self.init_layers()
        self.add_module('Attention', ChannelAttention(in_channels=out_channels)) if attention else None
        self.add_module('Activation', make_activation(activation)) if isinstance(activation, str) else None

    def init_layers(self): 
        for module in self: 
            module.apply(init_layers)
            
    def forward(self, input_data):
        for name, module in zip(self._modules, self):
            if 'Attention' not in name:
                input_data = module(input_data)
            else:
                input_data = module(input_data) + input_data
        return input_data


class DenseBlock(torch.nn.Sequential): 
    pass


class Down(torch.nn.Sequential):
    
    def __init__(self, channels):
        super(Down, self).__init__()
        self.add_module('Down Sample Max Pool', torch.nn.MaxPool2d(kernel_size=2))


class Up(torch.nn.Module):
    
    def __init__(self, channels, attention=False, interpolation=None, transpose=None, **kwargs):
        super(Up, self).__init__()
        assert any([interpolation, transpose]) and not all([interpolation, transpose])
        if isinstance(interpolation, str):
            assert transpose is None 
            self.up = torch.nn.Upsample(scale_factor=2, mode=interpolation, align_corners=True)
        elif isinstance(transpose, bool):
            assert interpolation is None
            self.up = torch.nn.ConvTranspose2d(in_channels=channels, out_channels=channels, 
                                               kernel_size=2, stride=2)

    def forward(self, lower_level, skip_connection):
        lower_level = self.up(lower_level)
        
        diff_x = skip_connection.shape[-1] - lower_level.shape[-1]
        diff_y = skip_connection.shape[-2] - lower_level.shape[-2]
        # pad the inputs
        lower_level = torch.nn.functional.pad(lower_level, [diff_x // 2, diff_x - diff_x // 2,
                                                            diff_y // 2, diff_y - diff_y // 2])
        
        lower_level = torch.cat([lower_level, skip_connection], dim=1)
        return lower_level

