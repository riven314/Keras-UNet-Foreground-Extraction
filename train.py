"""
FLEXIBLE PARAMETERS:
1. batch size
2. epochs

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
import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from optimizer import compile_unet
from get_data_gen import get_data_gen, DATA_AUG_ARGS

############# 1. SET UP KEY PARAMETERS #############
TRAIN_PATH = os.path.join()
VAL_PATH = os.path.join()
INPUT_SHAPE = None
BATCH_SIZE = 3
EPOCHS = 10

print('Train Path: {}'.format(TRAIN_PATH))
print('Val Path: {}'.format(VAL_PATH))
print('Input Size: {}'.format(INPUT_SHAPE))
print('Batch Size: {}'.format(BATCH_SIZE))
print('# Epochs: {}'.format(EPOCHS))


############## 2. SET UP MODEL AND DATA ##############
model = compile_unet(INPUT_SHAPE)

train_gen = None
val_gen = None
train_n = None
val_n = None


############## 3. SET UP TRAINING ##################
early_stop = EarlyStopping(monitor = 'val_loss', patience = 5, verbose = 1)
model_save = ModelCheckpoint('model-{epoch:03d}.h5', save_best_only = True, monitor = 'val_loss')
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 2, verbose = 1, epsilon = 1e-4)

print('Start training ...')
# start training
model.fit_generator(train_gen, steps_per_epoch = train_n // BATCH_SIZE,
                    validation_data = val_gen, validation_steps = max(val_n // BATCH_SIZE, 1),
                    epochs = EPOCHS,
                    callbacks = [early_stop, model_save, reduce_lr])
