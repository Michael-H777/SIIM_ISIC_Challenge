from packages import * 

from models.building_blocks import * 
from models.conv_net import *

from models.networks import * 


parser = argparse.ArgumentParser(description='templet train file')
parser.add_argument('--preprocess', type=bool, default=False, help='process training data using specified directory')
parser.add_argument('--start_lr', type=float, default=1e-2, help='initial learning rate, decay to end_learning_rate with cosine annihilation')
parser.add_argument('--end_lr', type=float, default=1e-5, help='ending learning rate')
parser.add_argument('--max_epoch', type=int, default=50, help='maximum epoch for training')
parser.add_argument('--epoch_samples', type=int, default=5_000, help='samples for each epochs')
parser.add_argument('--epoch_updates', type=int, default=100, help='allowed updates for each epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size when training')
parser.add_argument('--fine_tune_last', type=bool, default=False, help='fine tune last model?')
parser.add_argument('--model_pickle_path', type=str, default='', help='pickled model path, use with scheduler')
parser.add_argument('--log_path', type=str, default='/home/michael/ssd_cache/SMART/train_logs', help='train logs location')
parser.add_argument('--task_type', type=str, default='Denoise', help='Task type in train_log_path folder')
parser.add_argument('--GPU', type=str, default='0', help='specify which GPU to train on')
parser.add_argument('--debug', type=bool, default=False, help='set to debug mode')
options = parser.parse_args()

options.DDP = False


shape = (128,128)

model = classification_cuda(options=options, input_shape=shape)
data = torch.ones(16, 3, *shape)

model.set_input(data)
model.forward()

