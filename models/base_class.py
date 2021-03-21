import torch 
import math

import loss.perceptual as perceptual 
from abc import ABC, abstractmethod 




class base_model(ABC): 
    
    def __init__(self, loss=['l1']):
        self.train_loss = []
        self.test_loss = []
        self.loss_names = [name if isinstance(name, str) else name[0] for name in loss] 
        self.loss_functions = [make_loss(name) if isinstance(name,str) else name[1].cuda() for name in loss]
    
    
    def make_loss(name): 
        if name =='l1':
            loss_function = torch.nn.L1Loss()
        elif name == 'l2':
            loss_function = torch.nn.MSELoss() 
        elif name == 'ssim':
            pass 
        elif name == 'perceptual_vgg':
            loss_function = perceptual.VGG() 
        else:
            raise ValueError(f'loss function {name} not implemented')
        return loss_function.cuda() 
    
    ########################################################################
    # overwrite when using composite model 
    def compute_loss(self, record):
        loss = [loss_function(self.prediction, self.target) for loss_function in self.loss_functions]
        record.append([item.detach() for item in loss])
        return loss 
    
    def compute_params(self):
        params = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        return f'{params:_}'

    def forward(self):
        self.prediction = self.model.forward(self.input_data)
        return self.prediction
    
    def save_check_point(self, path):
        check_point = {'loss': self.loss_names, 
                       'epoch': self.current_epoch, 
                       'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(check_point, f'{path}/check_point.pth')
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), f'{path}/epoch_{self.current_epoch}.pth')
    
    def validataion(self):
        self.model.eval() 
        self.forward()
        self.compute_loss(self.test_loss)
    
    def take_step(self, batch_index):
        loss = self.compute_loss(self.train_loss)
        # BP the loss 
        total_loss = sum(loss)
        # only update in desired batch index 
        total_loss.backward() 
        if batch_index in self.update_at_steps:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True) # this does the same thing as model.zero_grad()
            self.model.train()
    # overwrite when using composite model 
    ########################################################################
    
    def check_attrs(self):
        attr_checks = ['name', 'loss_names', 'loss_functions', 'optimizer', 'model']
        for attr_name in attr_checks:
            assert hasattr(self, attr_name)
    
    def compute_avg_loss(self, loss): 
        # this moves all the detached things from GPU to CPU
        loss = [[column if isinstance(column, int) else column.cpu().numpy() for column in row] for row in loss]
        # this turns [[row1, row1, ...], 
        #             [row2, row1, ...], 
        #             [row3, row1, ...], ...] 
        # into [col1, col2, col3, ...]
        loss = [sum(loss[row][column] for row in range(len(loss)))/len(loss) for column in range(len(loss[0]))]
        return [round(column, 5) for column in loss]

    def cosine_annihilation(self):
        return self.min_lr + (self.start_lr - self.min_lr) * (1 + math.cos(math.pi * self.current_epoch / self.max_epoch)) / 2

    def gather_train_loss(self):
        return self.compute_avg_loss(self.train_loss)
    
    def gather_test_loss(self):
        return self.compute_avg_loss(self.test_loss)
    
    def set_input(self, data):
        self.input_data = data[:,0:1,:,:].cuda() 
        self.target = data[:,-1:,:,:].cuda()
    
    def set_requires_grad(self, models, requires_grad): 
        if not isinstance(models, list): 
            models = [models]
                
        for model in models:
            for param in model.parameters():
                param.requires_grad = requires_grad
    
    def update_lr(self, current_epoch):
        # clear these two for new epoch 
        self.train_loss = []
        self.test_loss = [] 
        self.current_epoch = current_epoch
        current_lr = self.cosine_annihilation()
        for param in self.optimizer.param_groups:
            param['lr'] = current_lr
        return current_lr
    
    def update_options(self, options):
        self.start_lr = options.start_lr 
        self.min_lr = options.end_lr 
        self.max_epoch = options.max_epoch 
                                                # total batches    # how many batches between updates   # which batch its on when updating 
        self.update_at_steps = set(int(options.epoch_samples/options.batch_size/options.epoch_updates * step) for step in range(1, options.epoch_updates))
    
    
class base_optimizer: 
    
    def __init__(self, optimizers): 
        assert isinstance(optimizers, list)
        
        self.optimizers = optimizers 
        
    @property
    def param_groups(self):
        # an iterator to yield all the param groups in all the optimizers 
        # to keep consistent behavior with torch.optim.xxx.param_groups

        for each in self.optimizers: 
            
            for param in each.param_groups:
                yield param
        
    def load_state_dict(self, dictionaries):
        assert len(self.optimizers) == dictionaries
        [each.load_state_dict(dictionary) for each, dictionary in zip(self.optimizers, dictionaries)]
        return None 
        
    def state_dict(self):
        return [each.state_dict() for each in self.optimizers]
    