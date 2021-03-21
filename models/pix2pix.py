import torch 

from models.unet import d_unet
from models.building_blocks import *
from models.base_class import base_model, base_optimizer


class GANLoss(torch.nn.Module): 
    
    def __init__(self, mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = mode 
        self.loss = torch.nn.L1Loss() if mode == 'lsgan' else None

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)    

    def __call__(self, prediction, target_is_real):
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


class NLayerDiscriminator(torch.nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=torch.nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()

        use_bias = False
        kw = 4
        padw = 1
        sequence = [torch.nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), torch.nn.GELU()]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                torch.nn.GELU()
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            torch.nn.GELU()
        ]

        sequence += [torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class Pix2Pix_cuda(base_model): 
    
    def __init__(self, *, loss=['l1', 'l2', 'perceptual_vgg'], optimizer=torch.optim.Adam, name=None, **kwargs):
        
        self.name = 'Pix2Pix' if name is None else name
        
        generator = d_unet(in_channels=1, out_channels=1, layers=4, channel_growth=32, **kwargs)
        #discriminator = NLayerDiscriminator(2, 64, 5)
        discriminator = d_unet(in_channels=2, out_channels=1, layers=4, **kwargs)
        
        self.generator = torch.nn.DataParallel(generator).cuda() 
        self.discriminator = torch.nn.DataParallel(discriminator).cuda()  
        
        self.gen_optimizer = optimizer(self.generator.parameters())
        self.dis_optimizer = optimizer(self.discriminator.parameters())
        
        self.optimizer = base_optimizer([self.gen_optimizer, self.dis_optimizer])
        
        self.gen_loss_func = [base_model.make_loss(name) for name in loss] 
        self.gen_loss_weight = [1, 0.3, 0.5]
        self.dis_loss_func = [GANLoss('wgangp', 0.9, 0.1).cuda()]
        self.loss_names = loss + ['generator_GANLoss', 'discriminator_GANLoss']
        
        self.train_loss = []
        self.test_loss = [] 

    def _compute_discriminator_loss(self):
        # fake
        fake_AB = torch.cat((self.real_A, self.fake_B), 1) 
        pred_fake = self.discriminator(fake_AB.detach())
        loss_D_fake = [loss_func(pred_fake, False) for loss_func in self.dis_loss_func]
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.discriminator(real_AB)
        loss_D_real = [loss_func(pred_real, True) for loss_func in self.dis_loss_func]
        # combine loss and calculate gradients
        loss_D = sum(loss_D_fake) + sum(loss_D_real)
        return [loss_D]
        
    def _compute_generator_loss(self):
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        loss_G_GAN = [loss_func(pred_fake, True) for loss_func in self.dis_loss_func]
        # Second, G(A) = B
        loss_G = [loss_func(self.fake_B, self.real_B) for loss_func in self.gen_loss_func]
        return loss_G + loss_G_GAN
        
    def forward(self):
        self.fake_B = self.real_A - self.generator(self.real_A)
        self.prediction = self.fake_B
    
    def set_input(self, input_data): 
        self.real_A = input_data[: , 0:1 , : , :].cuda()
        self.real_B = input_data[: , -1: , : , :].cuda()
        self.input_data, self.target = self.real_A, self.real_B
    
    def take_step(self, batch_index):
        self.generator.train()
        self.discriminator.train()
        
        # update D 
        self.forward()
        self.set_requires_grad(self.discriminator, True) 
        discriminator_loss = self._compute_discriminator_loss()
        bp_loss = sum(discriminator_loss)
        bp_loss.backward()
        if batch_index in self.update_at_steps:
            self.dis_optimizer.step()
            self.dis_optimizer.zero_grad(set_to_none=True)
        
        # update G
        self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
        generator_loss = self._compute_generator_loss()
        bp_loss = sum(generator_loss)
        bp_loss.backward()
        if batch_index in self.update_at_steps:
            self.gen_optimizer.step()
            self.gen_optimizer.zero_grad(set_to_none=True)
        
        step_loss = [item if isinstance(item, int) else item.detach() for item in generator_loss + discriminator_loss]
        self.train_loss.append(step_loss)
        
    def validation(self):
        self.generator.eval()
        self.discriminator.eval()
        
        self.forward()
        generator_loss = self._compute_generator_loss()
        discriminator_loss = self._compute_discriminator_loss()
        
        loss = [item if isinstance(item, int) else item.detach() for item in generator_loss + discriminator_loss]
        self.test_loss.append(loss)
        
    def save_check_point(self, path):
        check_point = {'loss': self.loss_names, 
                       'epoch': self.current_epoch, 
                       'generator': self.generator.state_dict(),
                       'discriminator': self.discriminator.state_dict(),
                       'optimizer': self.optimizer.state_dict()}
        torch.save(check_point, f'{path}/check_point.pth')
    
    def save_model(self, path):
        model = {'generator': self.generator.state_dict(),
                 'discriminator': self.discriminator.state_dict()}
        torch.save(model, f'{path}/epoch_{self.current_epoch}.pth')
    
    def compute_params(self):
        params = 0
        for model in [self.generator, self.discriminator]:
            params += sum(param.numel() for param in model.parameters() if param.requires_grad)
        return f'{params:_}'
    