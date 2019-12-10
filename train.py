"""
FLEXIBLE PARAMETERS:
1. batch size
2. epochs

TO BE SCALED UP LATER:
1. add tensorboard to monitor metrics and sample image results
2. set up module specifically for configuration
3. separate different exp. by folder
4. build script for evaluation

REMARKS:
1. for model capacity testing, train loss = 0.02-0.03, dice = 0.88, jaccard = 20 (disable reduce lr, data aug, early stop and dropout)

REFERENCE
1. setting learning rate adaptive to loss improvement
    - https://stackoverflow.com/questions/54764012/keras-i-want-to-decay-learning-rate-when-val-acc-stops-improving
2. adaptive lr + early stop training + save best model
    - https://stackoverflow.com/questions/48285129/saving-best-model-in-keras
3. documentation on callbacks: https://keras.io/callbacks/
4. multiple GPU training: https://datascience.stackexchange.com/questions/23895/multi-gpu-in-keras
"""
import os
import sys
import argparse

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from optimizer import compile_unet
from get_data_gen import DATA_AUG_ARGS, generate_data
from data import testGenerator, saveResult


############# 1. SET UP KEY PARAMETERS #############
# parse cmd arguments
parser = argparse.ArgumentParser(description = 'load in configurations')
parser.add_argument('--train_path', type = str, help = 'specify path to training dir (dir with image, label folder), jpg as imaage, png as mask')
parser.add_argument('--val_path', type = str, help = 'specify path to val dir (dir with image, label folder), jpg as imaage, png as mask')
parser.add_argument('--test_path', type = str, help = 'specify path to test dir (with image files .jpg)')
parser.add_argument('--vis_path', type = str, default = None, help = 'specify path to visualize the test set result')
parser.add_argument('--input_shape', nargs = '+', type = int, default = [480, 640], help = 'image size for model feed, [height, width]')
parser.add_argument('--bs', type = int, default = 3, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs for training')
parser.add_argument('--dropout', type = float, default = 0.5, help = 'dropout rate')
parser.add_argument('--scale', type = float, default = 0.5, help = 'scale of UNet (e.g. 0.5 mean halves the parameters)')
parser.add_argument('--gpu_n', type = int, default = 1, help = 'no. of GPU, if gpu_n>1, activate multi-GPU training')
parser.add_argument('--no_aug', action = 'store_true', help = 'whether activate data augmentation')
parser.add_argument('--no_early_stop', action = 'store_true', help = 'whether activate early stopping')
parser.add_argument('--no_save_model', action = 'store_true',  help = 'whether save model weights')
parser.add_argument('--no_reduce_lr', action = 'store_true',  help = 'whether set adaptive learning rate')

args = parser.parse_args()
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
TEST_PATH = args.test_path
VIS_PATH = args.vis_path
# (height, width)
INPUT_SHAPE = (args.input_shape[0], args.input_shape[1])
BATCH_SIZE = args.bs
EPOCHS = args.epochs
DROPOUT = args.dropout
SCALE = args.scale
GPU_N = args.gpu_n
IS_DATA_AUG = not args.no_aug
IS_EARLY_STOP = not args.no_early_stop
IS_SAVE_MODEL = not args.no_save_model
IS_REDUCE_LR = not args.no_reduce_lr

# sanity check
assert os.path.isdir(TRAIN_PATH), '[PATH ERROR] training path not exist'
assert os.path.isdir(VAL_PATH), '[PATH ERROR] val path not exist'

print('Train Path: {}'.format(TRAIN_PATH))
print('Val Path: {}'.format(VAL_PATH))
print('Test Path: {}'.format(TEST_PATH))
print('Vis Path: {}'.format(VIS_PATH))
print('Input Size: {}'.format(INPUT_SHAPE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('No. of Epochs: {}'.format(EPOCHS))
print('Dropout Rate: {}'.format(DROPOUT))
print('Scale: {}'.format(SCALE))
print('No. GPU: {}'.format(GPU_N))
print('Aug Mode: {}'.format(IS_DATA_AUG))
print('Early Stop: {}'.format(IS_EARLY_STOP))
print('Save Model: {}'.format(IS_SAVE_MODEL))
print('Reduce LR: {}'.format(IS_REDUCE_LR))


############## 2. SET UP MODEL AND DATA ##############
# attach channel 1 in last
model = compile_unet(INPUT_SHAPE + (1,), 
                     scale = SCALE, 
                     dropout = DROPOUT,
                     gpu_n = GPU_N)

if IS_DATA_AUG:
    DATA_AUG_ARGS = None

train_gen, train_n = generate_data(batch_size = BATCH_SIZE, 
                                   train_path = TRAIN_PATH, 
                                   aug_dict = DATA_AUG_ARGS,
                                   target_size = INPUT_SHAPE)

val_gen, val_n = generate_data(batch_size = BATCH_SIZE,
                               train_path = VAL_PATH,
                               aug_dict = DATA_AUG_ARGS,
                               target_size = INPUT_SHAPE)


############## 3. SET UP TRAINING ##################
if not os.path.isdir('weights'):
    os.mkdir('weights')


call_back_ls = []
if IS_EARLY_STOP:
    early_stop = EarlyStopping(monitor = 'loss', 
                               patience = 8, 
                               verbose = 1)
    call_back_ls.append(early_stop)
if IS_SAVE_MODEL:
    model_save = ModelCheckpoint(os.path.join('weights', 'model-{epoch:03d}.h5'), 
                                 save_best_only = True, 
                                 monitor = 'val_loss')
    call_back_ls.append(model_save)
if IS_REDUCE_LR:
    reduce_lr = ReduceLROnPlateau(monitor = 'loss', 
                                  factor = 0.2, 
                                  patience = 5, 
                                  verbose = 1, 
                                  epsilon = 1e-4)
    call_back_ls.append(reduce_lr)
if call_back_ls == []:
    call_back_ls = None


print('Start training ...')
# start training
model.fit_generator(train_gen, steps_per_epoch = train_n // BATCH_SIZE,
                    validation_data = val_gen, validation_steps = max(val_n // BATCH_SIZE, 1),
                    epochs = EPOCHS,
                    callbacks = call_back_ls)


############### 4. TEST EVALUATION ###################
print('Test Set Evaluation')
test_f_ls = [i for i in os.listdir(os.path.join(TEST_PATH)) if i.endswith('.jpg')]
test_n = len(test_f_ls)
test_gen = testGenerator(TEST_PATH, 
                         num_image = 1,
                         target_size = INPUT_SHAPE)
# shape: (steps, height, width, 1)
results = model.predict_generator(test_gen, 
                                  steps = test_n, 
                                  verbose = 1)
if VIS_PATH is not None: saveResult(VIS_PATH, results, test_f_ls)

