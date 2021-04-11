import json 
import torch 

import numpy as np 

from abc import ABC, abstractmethod 


class model_wrapper: 
    
    def __init__(self, *, options, model, loss, optimizer=torch.optim.Adam, 
                 scheduler=torch.optim.lr_scheduler.OneCycleLR):
        assert isinstance(loss, dict)
        self.loss = loss if loss else {}
        self.model = model 
        self.optimizer_class = optimizer 
        self.scheduler_class = scheduler 
        self.update_scheduler(options, last_epoch=-1)
        
        self.info = {'loss': [item.__class__.__name__ for item in self.loss.values()], 
                     'model': self.model.class_name,
                     'size': f'{sum(param.numel() for param in self.model.parameters() if param.requires_grad):_}', 
                     'optimizer': self.optimizer.__class__.__name__, 
                     'scheduler': self.scheduler.__class__.__name__}
    
    def forward(self, data):
        return self.model.forward(data)
    
    def update_scheduler(self, options, last_epoch):
        if last_epoch < 0:
            last_epoch = -1 
        else:
            last_epoch=last_epoch*options.epoch_updates
        self.optimizer = self.optimizer_class(self.model.parameters(), lr=options.max_lr)
        self.scheduler = self.scheduler_class(self.optimizer, max_lr=options.max_lr, epochs=options.max_epoch, 
                                              steps_per_epoch=options.epoch_updates, last_epoch=last_epoch)
        return None


class base_model(ABC): 
    
    @abstractmethod
    def train(self):
        pass 

    @abstractmethod
    def test(self):
        pass 

    def __init__(self, options):
        self.current_lr = options.max_lr
        self.update_rules(options)
        self.structure = {}
        self.test_loss = []
        self.train_loss = []
        self.scaler = torch.cuda.amp.GradScaler()

    ########################################################################################
    # save and load checkpoint/inference
    def save_check_point(self, path, epoch):
        package = {'epoch': epoch}
        # check point need to save model and optimizer state dict
        for model_name, model_wrapper in self.structure.items():
            package[model_name] = {}
            package[model_name]['model'] = model_wrapper.model.state_dict()
            package[model_name]['optimizer'] = model_wrapper.optimizer.state_dict()
        torch.save(package, path)
        return None 

    def load_check_point(self, path):
        state_dict = torch.load(path)

        for model_name in self.structure.keys():
            self.structure[model_name].model.load_state_dict(state_dict[model_name]['model'])
            self.structure[model_name].optimizer.load_state_dict(state_dict[model_name]['optimizer'])
        return state_dict['epoch'] + 1

    def save_inference(self, path):
        package = {model_name: model_wrapper.model.state_dict() for model_name, model_wrapper in self.structure.items()}
        torch.save(package, path)
        return None 

    def load_inference(self, path):
        state_dict = torch.load(path)
        assert len(self.structure) == len(state_dict)
        assert set(self.structure.keys()) == set(state_dict.keys())

        for model_name, state in state_dict.items():
            self.structure[model_name].model.load_state_dict(state)
        return None
    # save and load checkpoint/inference
    ########################################################################################
    
    def compile_image(self, patches):
        # get gap dynamically to accomondate for 2d and 3d
        shape = list(patches[0].shape)
        shape[-1] = 20 
        gap = np.zeros(tuple(shape))
        
        output = []
        for index, patch in enumerate(patches, 1):
            package = [patch, gap] if index != len(patches) else [patch]
            output.extend(package)
        
        image = np.concatenate(output, axis=len(output[0].shape)-1)
        image = np.float32(image)
        return image
        
    def compute_avg_loss(self, loss): 
        # this moves all the detached things from GPU to CPU
        loss = [[column if isinstance(column, (int, float)) else column.detach().cpu().numpy() for column in row] for row in loss]
        
        # this turns [[row1, row1, ...], 
        #             [row2, row1, ...], 
        #             [row3, row1, ...], ...] 
        # into [col1, col2, col3, ...]
        loss = [sum(loss[row][column] for row in range(len(loss)))/len(loss) for column in range(len(loss[0]))]
        return [round(column, 5) for column in loss]

    def compute_params(self):
        params = [sum(param.numel() for param in model_wrapper.model.parameters() if param.requires_grad) for model_wrapper in self.structure.values()]
        return f'{sum(params):_}'

    def move_cpu(self, data):
        return data[0,0].detach().cpu().numpy().astype(np.float32)

    def move_model(self, model):
        if self.DDP: 
            result = torch.nn.parallel.DistributedDataParallel(model.cuda())
        else:
            result = torch.nn.DataParallel(model).cuda()
        result.class_name = model.__class__.__name__
        return result 

    def gather_loss(self, loss_type):
        record = self.train_loss if loss_type=='train' else self.test_loss 
        return self.compute_avg_loss(record)

    def scheduler_step(self):
        for model_wrapper in self.structure.values():
            if model_wrapper.scheduler : 
                model_wrapper.scheduler.step()
                
        return model_wrapper.optimizer.param_groups[0]['lr'] 
    
    def set_input(self, data):
        self.input_data = data[:,0:1].cuda(non_blocking=True) 
        self.target = data[:,-1:].cuda(non_blocking=True)
        return None 

    def set_requires_grad(self, model_wrapper, requires_grad):
        for param in model_wrapper.model.parameters():
            param.requires_grad = requires_grad
        return None

    def set_to_train(self):
        self.test_loss = []
        self.train_loss = []
        [model_wrapper.model.train() for model_wrapper in self.structure.values()]
        return None

    def set_to_test(self):
        [model_wrapper.model.eval() for model_wrapper in self.structure.values()]
        return None 

    def update_rules(self, options):
        self.DDP = options.DDP
        self.options = options
        self.update_at_batch = set(int(options.epoch_samples/options.batch_size/options.epoch_updates * step) for step in range(1, options.epoch_updates))
        return None 

    def update_scheduler(self, options, epoch):
        [model_wrapper.update_scheduler(options, epoch-1) for model_wrapper in self.structure.values()]
        return None 

    def __repr__(self):
        info = {name:module.info for name, module in self.structure.items()}
        return json.dumps(info, indent=4)
    
    __str__ = __repr__ 
    
    