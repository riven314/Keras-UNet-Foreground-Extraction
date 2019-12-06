"""
checking schedule
1. model output shape
2. sample check number of channels
3. check dropout rate (12, 16)

REFERENCE:
1. converting tensor to numpy: https://stackoverflow.com/questions/34097281/how-can-i-convert-a-tensor-into-a-numpy-array-in-tensorflow
"""
import os
import sys
import itertools

MODULE_PATH = os.path.join(os.getcwd(), '..')
sys.path.append(MODULE_PATH)

import tensorflow as tf
from keras.layers import Input
from unet import unet

# (height, width, channel #)
INPUT_SIZES = [(256, 256, 1), (480, 640, 1)]
# (scale, corresponding # channels)
SCALES_N_CHANNELS = [(2, 128), (1, 64), (0.5, 32)] 
DROPOUTS = [0.3, 0.5]


def check_dropout(model, dropout_rate):
    """
    input:
        model -- keras model object
        dropout_rate -- float, ground truth dropout rate
    """
    sess = tf.Session()
    with sess.as_default():
        rate1 = model.layers[12].rate
        rate2 = model.layers[16].rate
        assert dropout_rate == rate1 == rate2, 'WRONG DROPOUT RATE'
    print('dropout test pass!')


def check_channel(model, channel_n):
    """
    input:
        channel_n: ground truth channel number
    """
    n1 = model.layers[1].output.get_shape().as_list()[-1]
    n2 = model.layers[-3].output.get_shape().as_list()[-1]
    assert channel_n == n1 == n2, 'WRONG CHANNEL NUMBER'
    print('number of channel pass!')


def check_output_shape(model, output_shape):
    """
    input:
        output_shape -- tuple, (height, width)
    """
    h = model.layers[-1].output.get_shape().as_list()[1]
    w = model.layers[-1].output.get_shape().as_list()[2]
    assert output_shape[0] == h, 'WRONG HEIGHT FOR MODEL OUTPUT'
    assert output_shape[1] == w, 'WRONG WIDTH FOR MODEL OUTPUT'
    print('model output shape pass!')


if __name__ == '__main__':
    config_gen = itertools.product(INPUT_SIZES, SCALES_N_CHANNELS, DROPOUTS)
    for i in config_gen:
        input_size, scale_n_channel, dropout = i
        scale, channel_n = scale_n_channel
        print('\n### input size = {}, scale = {}, expected channel #: {}, dropout = {}'\
              .format(input_size, scale, channel_n, dropout))
        model = unet(input_shape = input_size, scale = scale, dropout = dropout)
        check_channel(model, channel_n)
        check_dropout(model, dropout)
        check_output_shape(model, input_size[:2])
    print('all test cases pass!')
