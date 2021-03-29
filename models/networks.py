import torch 
from models.base_class import * 
from models.building_blocks import *

from models.unet import * 
from models.conv_net import * 


class conv_net_cuda(base_model):

    def __init__(self, options, name=None, use_dense=False, **kwargs):
        super().__init__(options)
        
        self.name = f'{"" if use_dense else "d_"}conv_net' if name is None else name
        self.loss_names = ['l1']
        loss = {'l1': loss_types['l1']().cuda()}
        model = d_conv_net(**kwargs) if use_dense else conv_net(**kwargs)
        model = torch.nn.DataParallel(model).cuda() if not self.DDP else torch.nn.parallel.DistributedDataParallel(model.cuda())
        self.structure['conv_model'] = model_wrapper(options=options, model=model, loss=loss)
        
    def compute_loss(self):
        loss = [loss(self.input_data, self.prediction) for loss in self.structure['conv_model'].loss.values()]
        return loss 

    def train(self, batch_index):
        self.prediction = self.structure['conv_model'].forward(self.input_data)
        loss = self.compute_loss()
        total_loss = sum(loss) 
        total_loss.backward()
        
        self.train_loss.append([item.detach() for item in loss])
        if batch_index in self.update_at_batch:
            self.structure['conv_model'].optimizer.step()
            self.structure['conv_model'].optimizer.zero_grad()
        return None 

    def test(self):
        self.prediction = self.structure['conv_model'].forward(self.input_data)
        loss = self.compute_loss()
        
        self.test_loss.append([item.detach() for item in loss])
        return self.move_cpu(self.input_data), self.move_cpu(self.prediction), self.move_cpu(self.target)


class d_unet_cuda(base_model):
    
    def __init__(self, options, name=None, **kwargs):
        
        super().__init__(options)
        self.name = 'Dense_UNet' if name is None else name 
        self.loss_names = ['l1']
        loss = {'l1': loss_types['l1']().cuda()}
        model = d_unet(**kwargs)
        model = torch.nn.DataParallel(model).cuda() if not self.DDP else torch.nn.parallel.DistributedDataParallel(model.cuda())
        self.structure['Dense_Unet'] = model_wrapper(options=options, model=model, loss=loss, name='D_UNet')         
        
    def compute_loss(self):
        loss = [loss(self.input_data, self.prediction) for loss in self.structure['Dense_Unet'].loss.values()]
        return loss 

    def train(self, batch_index):
        self.prediction = self.structure['Dense_Unet'].forward(self.input_data)
        loss = self.compute_loss()
        total_loss = sum(loss) 
        total_loss.backward()
        
        self.train_loss.append([item.detach() for item in loss])
        if batch_index in self.update_at_batch:
            self.structure['Dense_Unet'].optimizer.step()
            self.structure['Dense_Unet'].optimizer.zero_grad()
        return None 

    def test(self):
        self.prediction = self.structure['Dense_Unet'].forward(self.input_data)
        loss = self.compute_loss()
        
        self.test_loss.append([item.detach() for item in loss])
        return self.move_cpu(self.input_data), self.move_cpu(self.prediction), self.move_cpu(self.target)

