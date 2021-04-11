from configs import *
from packages import * 

import data 
from train import train 
from copy import deepcopy 

parser = argparse.ArgumentParser(description='train file')
parser.add_argument('--preprocess', type=bool, default=False, help='process training data using specified directory')
parser.add_argument('--max_lr', type=float, default=1e-2, help='initial learning rate, decay to end_learning_rate with cosine annihilation')
parser.add_argument('--min_lr', type=float, default=1e-3, help='ending learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch for training')
parser.add_argument('--epoch_samples', type=int, default=5_000, help='samples for each epochs')
parser.add_argument('--epoch_updates', type=int, default=100, help='allowed updates for each epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size when training')
parser.add_argument('--fine_tune_last', type=bool, default=False, help='fine tune last model?')
parser.add_argument('--model_pickle_path', type=str, default='', help='pickled model path, use with scheduler')
parser.add_argument('--log_path', type=str, default='/home/michael/ssd_cache/SMART/train_logs', help='train logs location')
parser.add_argument('--task_type', type=str, default='Segmentation', help='Task type in train_log_path folder')
parser.add_argument('--GPU', type=str, default='0', help='specify which GPU to train on')
parser.add_argument('--debug', type=bool, default=False, help='set to debug mode')
options = parser.parse_args()


def dump_pickle(model, path):
    with open(path, 'wb') as fileout:
        pickle.dump(model, fileout)
    return None 


def main():

    if 'win' in sys.platform.lower():
        options.log_path = 'D:/Data/SMART/train_logs'
    if options.debug: 
        torch.autograd.set_detect_anomaly(True)
        options.max_epoch = 2
        options.epoch_samples = 40
        options.epoch_updates = 10
        options.batch_size = 4 
    else:
        warnings.filterwarnings("ignore")
    
    if len(options.GPU) > 1:
        world_size = len(options.GPU.split(','))
        options.DDP = True 
    else:
        options.DDP = False

    from models.candidate import segnet_fast, unet_fast, d_unet_fast, segnet_hq, unet_hq, d_unet_hq, xl_d_unet_hq, xl_d_unet_fast
    to_train = [xl_d_unet_hq, xl_d_unet_fast, d_unet_fast, unet_fast, unet_hq, ]
    
    line_gap = '-' * line_width
    for index, model_class in enumerate(to_train, 1):
        option_used = deepcopy(options)
        
        option_used.schedule = f'[{index}/{len(to_train)}]'
        model = model_class(options=option_used)
        path = f'D:/Data/SMART/schedule/{model.__class__.__name__}.pickle'
        dump_pickle(model, path)
        option_used.model_pickle_path = path 
        
        line = f' scheduler on {option_used.schedule} {model.__class__.__name__} '
        print(f'\n{line_gap}\n{line:-^{line_width}}\n{line_gap}\n')
        if option_used.DDP:
            torch.multiprocessing.spawn(train, nprocs=world_size, args=(world_size, option_used))
        else:
            train(option_used.GPU, 0, option_used)


if __name__ == '__main__':
    
    if options.preprocess:
        data.make_model_data_on_stack() 
    
    if not os.path.isdir('D:/Data/SMART/schedule/'):
        os.mkdir('D:/Data/SMART/schedule/')
        
    name = 'ISIC_0052212'
    
    pydicom
