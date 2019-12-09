import os
import argparse

parser = argparse.ArgumentParser(description = 'load in configurations')
parser.add_argument('--train_path', type = str, help = 'specify path to training dir (dir with image, label folder)')
parser.add_argument('--val_path', type = str, help = 'specify path to val dir (dir with image, label folder)')
parser.add_argument('--input_shape', nargs = '+', type = int, default = [480, 640], help = 'image size for model feed, [height, width]')
parser.add_argument('--bs', type = int, default = 3, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs for training')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout rate')
parser.add_argument('--scale', type = float, default = 0.5, help = 'scale of UNet (e.g. 0.5 mean halves the parameters)')
parser.add_argument('--no_aug', action = 'store_true', help = 'whether activate data augmentation')
parser.add_argument('--no_early_stop', action = 'store_true', help = 'whether activate early stopping')
parser.add_argument('--no_save_model', action = 'store_true',  help = 'whether save model weights')
parser.add_argument('--no_reduce_lr', action = 'store_true',  help = 'whether set adaptive learning rate')

args = parser.parse_args()
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
# (height, width)
INPUT_SHAPE = (args.input_shape[0], args.input_shape[1])
BATCH_SIZE = args.bs
EPOCHS = args.epochs
DROPOUT = args.dropout
SCALE = args.scale
IS_DATA_AUG = not args.no_aug
IS_EARLY_STOP = not args.no_early_stop
IS_SAVE_MODEL = not args.no_save_model
IS_REDUCE_LR = not args.no_reduce_lr

# sanity check
assert os.path.isdir(TRAIN_PATH), '[PATH ERROR] training path not exist'
assert os.path.isdir(VAL_PATH), '[PATH ERROR] val path not exist'

print('Train Path: {}'.format(TRAIN_PATH))
print('Val Path: {}'.format(VAL_PATH))
print('Input Size: {}'.format(INPUT_SHAPE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('No. of Epochs: {}'.format(EPOCHS))
print('Dropout Rate: {}'.format(DROPOUT))
print('Scale: {}'.format(SCALE))
print('Aug Mode: {}'.format(IS_DATA_AUG))
print('Early Stop: {}'.format(IS_EARLY_STOP))
print('Save Model: {}'.format(IS_SAVE_MODEL))
print('Reduce LR: {}'.format(IS_REDUCE_LR))
