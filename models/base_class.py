import torch 
from loss import *
from abc import ABC, abstractmethod 


def make_loss(name): 
    if name =='l1':
        loss = torch.nn.L1Loss()
    elif name == 'l2':
        loss = torch.nn.L2Loss() 
    elif name == 'ssim':
        pass 
    elif name == 'vgg':
        pass 
    else:
        raise ValueError(f'loss function {name} not implemented')
    return loss.cuda() 


class base_model(ABC): 
    
    def __init__(self, loss=['l1']):
        self.train_loss = []
        self.test_loss = []
        self.loss_names = [name if isinstance(name, str) else name[0] for name in loss] 
        self.loss_functions = [make_loss(name) if isinstance(name,str) else name[1] for name in loss]
    
    def check_attrs(self):
        attr_checks = ['name', 'loss_names', 'loss_functions', 'optimizer', 'model']
        for attr_name in attr_checks:
            assert hasattr(self, attr_name)
    
    def compute_avg_loss(self, loss): 
        # this moves all the detached things from GPU to CPU
        loss = [[column.cpu().numpy() for column in row] for row in loss]
        # this turns [[row1], [row2], [row3], ...] into [col1, col2, col3, ...]
        loss = [sum(loss[row][column] for row in range(len(loss)))/len(loss) for column in range(len(loss[0]))]
        return [round(column, 5) for column in loss]
    
    def compute_params(self):
        return sum(param.numel() for param in self.model.parameters() if param.requires_grad)

    def compute_loss(self, record):
        loss = [loss_function(self.prediction, self.target) for loss_function in self.loss_functions]
        record.append([item.detach() for item in loss])
    
    def cosine_annihilation(self, current_epoch):
        return self.min_lr + (self.start_lr - self.min_lr) * (1 + math.cos(math.pi * self.current_epoch / self.max_epoch)) / 2
    
    def forward(self):
        self.prediction = self.model.forward(self.input)
        return self.prediction

    def gather_train_loss(self):
        return self.compute_avg_loss(self.train_loss)
    
    def gather_test_loss(self):
        return self.compute_avg_loss(self.test_loss)
    
    def save_check_point(self, path):
        check_point = {'loss': self.loss_names, 
                       'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict()}
        torch.save(check_point, f'{path}/check_point.pth')
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), f'{path}/model_{self.current_epoch}.pth')
    
    def set_input(self, data):
        self.input = data[:,0:1,:,:].cuda() 
        self.target = data[:,-1:,:,:].cuda()
    
    def take_step(self, batch_index):
        self.compute_loss(self.train_loss)
        # BP the loss 
        total_loss = sum(loss)
        total_loss.backward() 
        # only update in desired batch index 
        if batch_index in self.update_at_steps:
            self.optimizer.step()
            self.optimizer.zero_grad() # this does the same thing as model.zero_grad()
            self.train() 
    
    def test(self):
        self.compute_loss(self.test_loss)
    
    def update_lr(self, current_epoch):
        # clear these two for new epoch 
        self.train_loss = []
        self.test_loss = [] 
        self.current_epoch = current_epoch
        for param in self.optimizer.param_groups:
            param['lr'] = self.cosine_annihilation()
    
    def update_options(self, options):
        self.start_lr = options.start_learning_rate 
        self.min_lr = options.end_learning_rate 
        self.max_epoch = options.max_epoch 
                                                # total batches    # how many batches between updates   # which batch its on when updating 
        self.update_at_steps = set(int(options.epoch_samples/options.batch_size/options.epoch_updates * step) for step in range(1, options.epoch_updates))
    