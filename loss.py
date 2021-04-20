import torch 
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Accuracy:
    
    def __call__(self, prediction, target):
        prediction = [round(item) if item is not np.nan else -1 for item in prediction]
        return accuracy_score(target, prediction) 
    
class F1:
    
    def __call__(self, prediction, target):
        prediction = [round(item) for item in prediction]
        return f1_score(target, prediction) 
    
class Roc_Auc:
    
    def __call__(self, prediction, target): 
        return roc_auc_score(target, prediction) 
    

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 4
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class Perceptual(torch.nn.Module):
    
    # https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49
    
    def __init__(self):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg19_bn(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg19_bn(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg19_bn(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg19_bn(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, input_data, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        if input_data.shape[1] != 3:
            input_data = input_data.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input_data = (input_data-self.mean) / self.std
        target = (target-self.mean) / self.std
        
        loss = 0.0
        x = input_data
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss * 0.5