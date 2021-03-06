import torch 
from models.base_class import * 
from models.building_blocks import *

from models.encoder_decoder import * 

from loss import Accuracy, F1, Roc_Auc


class templet(base_model):
        
    def classify(self):
        skip, bottom = self.encode_result
        result = [torch.flatten(tensor, start_dim=1) for tensor in [skip, bottom]]
        result = torch.cat(result, dim=1)
        self.classify_result = self.structure['classifier'].forward(result)
        return None 
        
    def decode(self):
        self.decode_result = self.structure['decoder'].forward(self.encode_result)
        return None 
    
    def encode(self):
        self.encode_result = self.structure['encoder'].forward(self.input_data)
        return None 
        
    def image_loss(self):
        loss = [function(self.decode_result, self.decode_target) for function in self.structure['decoder'].loss.values()]        
        return loss 
    
    def classify_loss(self):
        loss = [self.structure['classifier'].loss['BCElogits'](self.classify_result, self.classify_target)]
        return loss
        
    def train(self, batch_index):
        
        if self.do_image:
            with torch.cuda.amp.autocast():
                self.encode()
                self.decode()

                loss = self.image_loss()
                
            self.scaler.scale(sum(loss)).backward()
            
            if batch_index in self.update_at_batch:
                self.scaler.unscale_(self.structure['encoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['encoder'].model.parameters(), 1)
                self.scaler.step(self.structure['encoder'].optimizer)
                self.structure['encoder'].optimizer.zero_grad()
            
                self.scaler.unscale_(self.structure['decoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['decoder'].model.parameters(), 1)
                self.scaler.step(self.structure['decoder'].optimizer)
                self.structure['decoder'].optimizer.zero_grad()
                
                self.scaler.update() 
                self.current_lr = self.scheduler_step()
                self.do_image = False 
            loss = loss + [0]
        else:        
            with torch.cuda.amp.autocast():
                
                self.encode()
                self.classify()

                loss = self.classify_loss()
                    
            self.scaler.scale(sum(loss)).backward()
            
            if batch_index in self.update_at_batch:
                self.scaler.unscale_(self.structure['encoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['encoder'].model.parameters(), 1)
                self.scaler.step(self.structure['encoder'].optimizer)
                self.structure['encoder'].optimizer.zero_grad()
            
                self.scaler.unscale_(self.structure['classifier'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['classifier'].model.parameters(), 1)
                self.scaler.step(self.structure['classifier'].optimizer)
                self.structure['classifier'].optimizer.zero_grad()
                
                self.scaler.update() 
                self.current_lr = self.scheduler_step()
                self.do_image = True 
            loss = [0 for _ in self.structure['decoder'].loss] + loss 
        self.train_loss.append(loss)
        return self.current_lr
    
    def test(self):         
        with torch.cuda.amp.autocast():
            self.encode()
            self.decode()
            self.classify()
        
        prediction = torch.nn.Sigmoid()(self.classify_result)
        loss = self.image_loss() + self.classify_loss()
            
        self.test_loss.append(loss)
        
        prediction = prediction.cpu().numpy().flatten()[0]
        target = self.classify_target.cpu().numpy().flatten()[0]

        return prediction, target

    def set_input(self, data):
        # batch * (R,G,B,Gray,Label) * H * W
        self.input_data = data[:,0:4,:,:].cuda()
        self.decode_target = data[:,0:4,:,:].cuda()
        classify_target = torch.mean(data[:,-1:,:,:], dim=[1,2,3])
        self.classify_target = torch.unsqueeze(classify_target, dim=1).cuda()
        return None 
    
    def gather_loss(self, loss_type, data=None):
        if loss_type == 'train':
            record = self.compute_avg_loss(self.train_loss)
            record.extend([0 for _ in self.eval_loss])
        else:
            record = self.compute_avg_loss(self.test_loss)
            prediction = [item[0] for item in data]
            target = [item[1] for item in data]
            record.extend([function(prediction, target) for function in self.eval_loss])
        return record
    
    
class l1_l2_classification(templet):  
    
    def __init__(self, *, options, input_shape, name=None): 
        
        super().__init__(options)
        self.name = self.__class__.__name__ if name is None else name 
        
        layers = 5
        channel_growth = 32
        
        model = encoder(in_channels=4, layers=layers, channel_growth=channel_growth, block_layers=4)
        skip_channels, bottom_channels = model.channels
        model = self.move_model(model)
        self.structure['encoder'] = model_wrapper(options=options, model=model, loss={})
        
        loss = {'l1': loss_types['l1']().cuda(), 
                'l2': loss_types['l2']().cuda()}
        
        model = decoder(out_channels=4, layers=layers, channel_growth=channel_growth)
        model = self.move_model(model)
        self.structure['decoder'] = model_wrapper(options=options, model=model, loss=loss)
        
        loss = {'BCElogits': loss_types['BCElogits']().cuda()}
        skip_size = int(input_shape[0] * input_shape[1] / (4**(layers-1)) * skip_channels)
        bottom_size = int(input_shape[0] * input_shape[1] / (4**layers) * bottom_channels)
        #print(f'{skip_channels=}, {bottom_channels=}, {skip_size=}, {bottom_size=}')
        model = FC_classifier(input_size=skip_size+bottom_size, output_size=1, FC_layers=2)
        model = self.move_model(model)
        self.structure['classifier'] = model_wrapper(options=options, model=model, loss=loss)
        
        self.loss_names = ['decoder_l1', 'decoder_l2', 'BCElogits', 'Accuracy', 'F1', 'ROC_AUC']
        self.eval_loss = [Accuracy(), F1(), Roc_Auc()]
        self.do_image = True 

    
class l1_l2_ssim_vgg_classification(templet):  
    
    def __init__(self, *, options, input_shape, name=None): 
        
        super().__init__(options)
        self.name = self.__class__.__name__ if name is None else name 
        
        layers = 5
        channel_growth = 32
        
        model = encoder(in_channels=4, layers=layers, channel_growth=channel_growth, block_layers=4)
        skip_channels, bottom_channels = model.channels
        model = self.move_model(model)
        self.structure['encoder'] = model_wrapper(options=options, model=model, loss={})
        
        loss = {'l1': loss_types['l1']().cuda(), 
                'l2': loss_types['l2']().cuda(), 
                'ssim': loss_types['ssim']().cuda()}
        self.perceptual = loss_types['perceptual']().cuda()
        
        model = decoder(out_channels=4, layers=layers, channel_growth=channel_growth)
        model = self.move_model(model)
        self.structure['decoder'] = model_wrapper(options=options, model=model, loss=loss)
        
        loss = {'BCElogits': loss_types['BCElogits']().cuda()}
        skip_size = int(input_shape[0] * input_shape[1] / (4**(layers-1)) * skip_channels)
        bottom_size = int(input_shape[0] * input_shape[1] / (4**layers) * bottom_channels)
        #print(f'{skip_channels=}, {bottom_channels=}, {skip_size=}, {bottom_size=}')
        model = FC_classifier(input_size=skip_size+bottom_size, output_size=1, FC_layers=2)
        model = self.move_model(model)
        self.structure['classifier'] = model_wrapper(options=options, model=model, loss=loss)
        
        self.loss_names = ['decoder_l1', 'decoder_l2', 'decoder_ssim', 'decoder_vgg_RGB', 'decoder_vgg_gray','BCElogits', 'Accuracy', 'F1', 'ROC_AUC']
        self.eval_loss = [Accuracy(), F1(), Roc_Auc()]
        self.do_image = True  
        
    def image_loss(self):
        loss = [function(self.decode_result, self.decode_target) for function in self.structure['decoder'].loss.values()]
        vgg_rgb = self.perceptual(self.decode_target[:,0:3], self.decode_result[:,0:3])
        vgg_gray = self.perceptual(self.decode_target[:,-1:], self.decode_result[:,-1:])
        return loss + [vgg_rgb, vgg_gray]
    
    def train(self, batch_index):
        
        if self.do_image:
            with torch.cuda.amp.autocast():
                self.encode()
                self.decode()

                loss = self.image_loss()
                
            self.scaler.scale(sum(loss)).backward()
            
            if batch_index in self.update_at_batch:
                self.scaler.unscale_(self.structure['encoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['encoder'].model.parameters(), 1)
                self.scaler.step(self.structure['encoder'].optimizer)
                self.structure['encoder'].optimizer.zero_grad()
            
                self.scaler.unscale_(self.structure['decoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['decoder'].model.parameters(), 1)
                self.scaler.step(self.structure['decoder'].optimizer)
                self.structure['decoder'].optimizer.zero_grad()
                
                self.scaler.update() 
                self.current_lr = self.scheduler_step()
                self.do_image = False 
            loss = loss + [0]
        else:        
            with torch.cuda.amp.autocast():
                
                self.encode()
                self.classify()

                loss = self.classify_loss()
                    
            self.scaler.scale(sum(loss)).backward()
            
            if batch_index in self.update_at_batch:
                self.scaler.unscale_(self.structure['encoder'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['encoder'].model.parameters(), 1)
                self.scaler.step(self.structure['encoder'].optimizer)
                self.structure['encoder'].optimizer.zero_grad()
            
                self.scaler.unscale_(self.structure['classifier'].optimizer)
                torch.nn.utils.clip_grad_norm_(self.structure['classifier'].model.parameters(), 1)
                self.scaler.step(self.structure['classifier'].optimizer)
                self.structure['classifier'].optimizer.zero_grad()
                
                self.scaler.update() 
                self.current_lr = self.scheduler_step()
                self.do_image = True 
            loss = [0 for _ in self.structure['decoder'].loss] + [0,0] + loss 
        self.train_loss.append(loss)
        return self.current_lr
    
    def test(self):         
        with torch.cuda.amp.autocast():
            self.encode()
            self.decode()
            self.classify()
        
        prediction = torch.nn.Sigmoid()(self.classify_result)
        loss = self.image_loss() + self.classify_loss()
            
        self.test_loss.append(loss)
        
        prediction = prediction.cpu().numpy().flatten()[0]
        target = self.classify_target.cpu().numpy().flatten()[0]

        return prediction, target
