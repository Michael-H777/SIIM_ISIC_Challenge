import torch 
from models.base_class import * 
from models.building_blocks import *

from models.encoder_decoder import * 


class classification_cuda(base_model):
    
    def __init__(self, *, options, input_shape, name=None): 
        
        super().__init__(options)
        self.name = 'Encoder-Decoder-Classifier' if name is None else name 
        
        layers = 4
        channel_growth = 4 # 8 won't work zzzz
        
        model = encoder(in_channels=3, layers=layers, channel_growth=channel_growth, group_norm=False)
        skip_channels, bottom_channels = model.channels
        model = self.move_model(model)
        self.structure['encoder'] = model_wrapper(options=options, model=model, loss={}, name='encoder')
        
        loss = {'l1': loss_types['l1'], 
                'l2': loss_types['l2']}
        model = decoder(out_channels=3, layers=layers, channel_growth=channel_growth, group_norm=False)
        model = self.move_model(model)
        self.structure['decoder'] = model_wrapper(options=options, model=model, loss=loss, name='decoder')
        
        loss = {'cross_entropy': loss_types['cross_entropy']}
        skip_size = int(input_shape[0] * input_shape[1] / (4**(layers-1)) * skip_channels)
        bottom_size = int(input_shape[0] * input_shape[1] / (4**layers) * bottom_channels)
        print(f'{skip_channels=}, {bottom_channels=}, {skip_size=}, {bottom_size=}')
        model = FC_classifier(input_size=skip_size+bottom_size, output_size=1, FC_layers=4)
        model = self.move_model(model)
        self.structure['classifier'] = model_wrapper(options=options, model=model, loss=loss, name='FC Classifier')
        
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
    
    def forward(self): 
        self.encode()
        self.decode()
        self.classify()
        return None 
        
    def train(self):
        pass 
    
    def test(self):
        self.encode()
        self.classify()
        return self.classify_result

    def set_input(self, data):
        # batch * (R,G,B,Label) * H * W
        self.input_data = data[:,0:3,:,:].cuda()
        self.decode_target = input_data
        classify_target = data[:,-1:,:,:]
        classify_target = torch.mean(classify_target, dim=[1,2,3])
        self.classify_target = torch.unsqueeze(classify_target, dim=1).cuda()
        return None 
    