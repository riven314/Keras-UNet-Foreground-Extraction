"""
FLEXIBLE PARAMETERS:
1. batch size
2. epochs

TO BE SCALED UP LATER:
1. add tensorboard to monitor metrics and sample image results
2. set up module specifically for configuration
3. separate different exp. by folder
4. build script for evaluation

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


############# 1. SET UP KEY PARAMETERS #############
# parse cmd arguments
parser = argparse.ArgumentParser(description = 'load in configurations')
parser.add_argument('--train_path', type = str, help = 'specify path to training dir (dir with image, label folder)')
parser.add_argument('--val_path', type = str, help = 'specify path to val dir (dir with image, label folder)')
parser.add_argument('--input_shape', type = tuple, default = (480, 640), help = 'image size for model feed, (height, width)')
parser.add_argument('--bs', type = int, default = 3, help = 'batch size')
parser.add_argument('--epochs', type = int, default = 1, help = 'number of epochs for training')

args = parser.parse_args()
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
# (height, width)
INPUT_SHAPE = args.input_shape
BATCH_SIZE = args.bs
EPOCHS = args.epochs

# sanity check
assert os.path.isdir(TRAIN_PATH), '[PATH ERROR] training path not exist'
assert os.path.isdir(VAL_PATH), '[PATH ERROR] val path not exist'

print('Train Path: {}'.format(TRAIN_PATH))
print('Val Path: {}'.format(VAL_PATH))
print('Input Size: {}'.format(INPUT_SHAPE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('No. of Epochs: {}'.format(EPOCHS))

############## 2. SET UP MODEL AND DATA ##############
# attach channel 1 in last
model = compile_unet(INPUT_SHAPE + (1,))

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

early_stop = EarlyStopping(monitor = 'val_loss', 
                           patience = 5, 
                           verbose = 1)
model_save = ModelCheckpoint(os.path.join('weights', 'model-{epoch:03d}.h5'), 
                             save_best_only = True, 
                             monitor = 'val_loss')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                              factor = 0.2, 
                              patience = 2, 
                              verbose = 1, 
                              epsilon = 1e-4)

print('Start training ...')
# start training
model.fit_generator(train_gen, steps_per_epoch = train_n // BATCH_SIZE,
                    validation_data = val_gen, validation_steps = max(val_n // BATCH_SIZE, 1),
                    epochs = EPOCHS,
                    callbacks = [early_stop, model_save, reduce_lr])
