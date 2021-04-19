import torch 
import numpy as np 

from loss import SSIM, Perceptual

loss_types = {'l1': torch.nn.L1Loss, 
              'l2': torch.nn.MSELoss, 
              'ssim': SSIM, 
              'perceptual': Perceptual, 
              'BCE': torch.nn.BCELoss, 
              'BCElogits': torch.nn.BCEWithLogitsLoss,
              'cross_entropy': torch.nn.CrossEntropyLoss}


def make_activation(function_type):
    if function_type == 'relu': 
        return torch.nn.ReLU(inplace=False)
    elif function_type == 'leaky_relu':
        return torch.nn.LeakyReLU(inplace=False)
    elif function_type == 'elu':
        return torch.nn.ELU(inplace=False)
    elif function_type == 'gelu':
        return torch.nn.GELU()
    else:
        raise NotImplementedError(f'activation type {function_type} undefined in code')


def init_layers(layer):
    layer_name = layer.__class__.__name__ 
    if layer_name.find('Conv') != -1: 
        torch.nn.init.kaiming_normal_(layer.weight.data, mode='fan_in')
    elif layer_name.find('BatchNorm') != -1: 
        # what are these numbers and where they came from?????????
        layer.weight.data.normal_(mean=0, std=np.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        torch.nn.init.constant_(layer.bias.data, 0.0)
    return None


class ChannelAttention2D(torch.nn.Module): 
    
    # https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf
    
    def __init__(self, channels, reduction=2):
        super().__init__()
        
        self.attention = torch.nn.Sequential(
            ConvLayer2D(in_channels=channels, out_channels=channels), 
            ConvLayer2D(in_channels=channels, out_channels=channels, activation=None), 
            torch.nn.AdaptiveAvgPool2d(1), 
            torch.nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0, bias=True), 
            torch.nn.GELU(),
            torch.nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0, bias=True), 
            torch.nn.Sigmoid()
        )
        
    def forward(self, input_data):
        residual = self.attention(input_data)
        return input_data + residual


class ChannelAttention3D(torch.nn.Module): 
    
    # https://openaccess.thecvf.com/content_ECCV_2018/papers/Yulun_Zhang_Image_Super-Resolution_Using_ECCV_2018_paper.pdf
    
    def __init__(self, channels, reduction=2):
        super().__init__()
        
        self.attention = torch.nn.Sequential(
            ConvLayer3D(in_channels=channels, out_channels=channels), 
            ConvLayer3D(in_channels=channels, out_channels=channels, activation=None), 
            torch.nn.AdaptiveAvgPool3d(1), 
            torch.nn.Conv3d(channels, channels//reduction, kernel_size=1, padding=0, bias=True), 
            torch.nn.GELU(),
            torch.nn.Conv3d(channels//reduction, channels, kernel_size=1, padding=0, bias=True), 
            torch.nn.Sigmoid()
        )
        
    def forward(self, input_data):
        residual = self.attention(input_data)
        return input_data + residual
    
    
class ConvLayer2D(torch.nn.Sequential): 
    
    def __init__(self, *, in_channels, out_channels, kernel_size=5, dilation=1, stride=1, padding=None,
                 padding_mode='reflect', group_norm=16, dropout=False, activation='gelu'):
        super().__init__()
        
        # calculate padding after dilation 
        if padding is None:
            dilated_space = (kernel_size // 2) * (dilation-1)
            dilated_kernel = kernel_size + dilated_space*2
            padding = (dilated_kernel - 1) // 2
            
        # Conv is must have; but BN, dropout, activation are dependent on model architecture, 
        # which is controlled by changing input param when buildind said model 
        self.add_module('Conv', torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                                kernel_size=kernel_size, stride=stride, bias=False, 
                                                dilation=dilation, padding=padding, padding_mode=padding_mode))
        # groupnorm should help with small batch size and accumulating gradient 
        # as the original groupnorm paper show, group of 16 per shows best performance 
        if isinstance(group_norm, int) and out_channels >= group_norm:
            norm = torch.nn.GroupNorm(num_groups=out_channels//group_norm, num_channels=out_channels)
        else:
            norm = torch.nn.BatchNorm2d(num_features=out_channels)
        self.add_module('Normalize', norm)
        self.add_module('Dropout', torch.nn.Dropout(dropout, inplace=True)) if isinstance(dropout, float) else None 
        self.init_layers()
        self.add_module('Activation', make_activation(activation)) if isinstance(activation, str) else None
        
    def init_layers(self): 
        for module in self: 
            module.apply(init_layers)

    def forward(self, input_data):
        for module in self:
            input_data = module(input_data)
        return input_data

    
class ConvLayer3D(torch.nn.Sequential): 
    
    def __init__(self, *, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, padding=None,
                 padding_mode='zeros', group_norm=16, dropout=False, activation='gelu'):
        super().__init__()
        
        # calculate padding after dilation 
        if padding is None:
            dilated_space = (kernel_size // 2) * (dilation-1)
            dilated_kernel = kernel_size + dilated_space*2
            padding = (dilated_kernel - 1) // 2
            
        # Conv is must have; but BN, dropout, activation are dependent on model architecture, 
        # which is controlled by changing input param when buildind said model 
        self.add_module('Conv', torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
                                                kernel_size=kernel_size, stride=stride, bias=False, 
                                                dilation=dilation, padding=padding, padding_mode=padding_mode))
        # groupnorm should help with small batch size and accumulating gradient 
        # as the original groupnorm paper show, group of 16 per shows best performance 
        if isinstance(group_norm, int) and out_channels >= group_norm:
            norm = torch.nn.GroupNorm(num_groups=out_channels//group_norm, num_channels=out_channels)
        else:
            norm = torch.nn.BatchNorm3d(num_features=out_channels)
        self.add_module('Normalize', norm)
        self.add_module('Dropout', torch.nn.Dropout(dropout, inplace=True)) if isinstance(dropout, float) else None 
        self.init_layers()
        self.add_module('Activation', make_activation(activation)) if isinstance(activation, str) else None
        
    def init_layers(self): 
        for module in self: 
            module.apply(init_layers)

    def forward(self, input_data):
        for module in self:
            input_data = module(input_data)
        return input_data
    
    
class DenseBlock(torch.nn.Module): 
    
    def __init__(self, *, in_channels, out_channels, channel_growth=16,
                 block_layers=3, block_attention=True, use_3d=False, **kwargs):
        super().__init__()
        
        if use_3d:
            conv_layer = ConvLayer3D
            attention_layer = ChannelAttention3D
        else:
            conv_layer = ConvLayer2D
            attention_layer = ChannelAttention2D
        
        self.conv = torch.nn.Sequential()
        layer_channels = [in_channels]
        block_layers -= block_attention*1
        for layer_number in range(block_layers):
            not_bottle_neck = layer_number != block_layers-1
            layer_out_channels = layer_channels[-1] + channel_growth if not_bottle_neck else out_channels 
            # make layer, set to bottleneck if last layer 
            layer = conv_layer(in_channels=sum(layer_channels), out_channels=layer_out_channels, **kwargs)
            self.conv.add_module(f'Dense Block Layer {layer_number+1}{"" if not_bottle_neck else " bottleneck"}', layer)
            layer_channels.append(layer_out_channels)
        self.attention = attention_layer(channels=out_channels) if block_attention else None
        
    def forward(self, input_data): 
        steps = [input_data]
        for module in self.conv: 
            new_step = module(torch.cat(steps, 1))
            steps.append(new_step)
        new_step = new_step + self.attention(new_step) if self.attention is not None else new_step
        return new_step 


class UNetLayer(torch.nn.Module):
    
    # idea from: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    # native implementation 
    
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """
    
    def __init__(self, *, channels, submodule, channel_growth=16, interpolation='bicubic', transpose=False, use_3d=False, **kwargs):
        
        super().__init__()
        assert any([interpolation, transpose]) and not all([interpolation, transpose])
        
        if use_3d: 
            pool_layer = torch.nn.MaxPool3d(kernel_size=(1,2,2))
            interpolation = 'trilinear'
            up_method = torch.nn.Upsample(scale_factor=(1,2,2), mode=interpolation, align_corners=True)
        else:
            pool_layer = torch.nn.MaxPool2d(2)
            up_method = torch.nn.Upsample(scale_factor=2, mode=interpolation, align_corners=True) if not transpose else \
                        torch.nn.ConvTranspose2d(center_channels, center_channels // 2, kernel_size=2, stride=2)
            
        
        center_channels = channels+channel_growth
        
        self.down = torch.nn.Sequential(
            pool_layer, 
            DenseBlock(in_channels=channels, out_channels=center_channels, use_3d=use_3d, **kwargs), 
        )
        
        self.submodule = submodule 
        
        self.up = torch.nn.Sequential(
            DenseBlock(in_channels=center_channels*2, out_channels=channels, use_3d=use_3d, **kwargs), 
            up_method
        )
        
    def forward(self, input_data):
        skip_connection = self.down(input_data)
        lower_layer = self.submodule(skip_connection)
        result = self.up(torch.cat([skip_connection, lower_layer], dim=1))
        return result
    
    
class Conv2dUp(torch.nn.Sequential):
    
    def __init__(self, channels):
        
        self.add_module('Conv K4S2 Up', ConvLayer2D(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=0))
        
        