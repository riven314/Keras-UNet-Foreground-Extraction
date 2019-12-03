"""
PROCEDURES:
1. intake a mask, concatenate into a batch size (e.g. 3)
2. compare itself on dice coef and jaccard distance (should give 100%)
"""
import os
import sys
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend as K
from metrics import dice_coef, jaccard_distance

MASK1_PATH = os.path.join(os.getcwd(), '..', '..', '..', 
                        'data', 'ready', 'train', 'label', '27441N_G.png')
MASK2_PATH = os.path.join(os.getcwd(), '..', '..', '..', 
                        'data', 'ready', 'train', 'label', '27453N_E.png')

# sanity check
assert os.path.isfile(MASK1_PATH), 'NO SUCH FILE!'
assert os.path.isfile(MASK2_PATH), 'NO SUCH FILE!'
mask_1 = cv2.imread(MASK1_PATH, cv2.IMREAD_GRAYSCALE)
mask_2 = cv2.imread(MASK2_PATH, cv2.IMREAD_GRAYSCALE)
mask_ls = [mask_1, mask_2]

# transform to tensor format
for i in range(len(mask_ls)):
    m = mask_ls[i]
    m = m / 255
    m = np.expand_dims(m, axis = 2)
    m = np.expand_dims(m, axis = 0)
    mask_ls[i] = m
mask = np.concatenate(mask_ls, axis = 0)

# convert to tensor and then compute two metrics
t = tf.convert_to_tensor(mask, np.float32)
sess = tf.InteractiveSession()
a = dice_coef(mask, mask).eval()
b = jaccard_distance(mask, mask).eval()
print('dice coefficient = {}'.format(a))
print('jaccard distance = {}'.format(b))
sess.close()

# unit test
assert a == 1., 'WRONG DICE COEFFICIENT'
assert b == 0., 'WRONG JACCARD DISTANCE'
print('PASS!')