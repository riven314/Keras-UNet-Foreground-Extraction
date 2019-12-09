import os
import sys

from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model

from unet import init_unet
from metrics import dice_coef, jaccard_distance
from data import *

def compile_unet(input_shape, scale = 0.5, dropout = 0.5, lr = 0.0001, gpu_n = 1):
    """
    do the following:
    1. init model
    2. set up metrics, loss and optimizer
    """
    unet = init_unet(input_shape, scale, dropout)
    # compile
    if gpu_n == 1:
        # single GPU
        unet.compile(optimizer = Adam(lr = lr),
                     loss = 'binary_crossentropy',
                     metrics = [dice_coef, jaccard_distance])
        return unet
    else:
        # multi gpu
        parallel_unet = multi_gpu_model(unet, gpus = gpu_n)
        parallel_unet.compile(optimizer = Adam(lr = lr),
                              loss = 'binary_crossentropy',
                              metrics = [dice_coef, jaccard_distance])
        return parallel_unet
        
    

