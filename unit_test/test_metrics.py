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
from metrics import dice_coef, jaccard_distance, fg_recall

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

# simpler case
m_true = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype = np.uint8)
m_true = np.expand_dims(m_true, axis = 2)
m_true = np.expand_dims(m_true, axis = 0)
m_true = np.concatenate([m_true, m_true], axis = 0)
m_pred = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype = np.uint8)
m_pred = np.expand_dims(m_pred, axis = 2)
m_pred = np.expand_dims(m_pred, axis = 0)
m_pred = np.concatenate([m_pred, m_pred], axis = 0)

# convert to tensor and then compute two metrics
t1 = tf.convert_to_tensor(mask, np.float32)
t_true = tf.convert_to_tensor(m_true, np.float32)
t_pred = tf.convert_to_tensor(m_pred, np.float32)
sess = tf.InteractiveSession()
a = dice_coef(t1, t1).eval()
b = jaccard_distance(t1, t1).eval()
c = fg_recall(t_true, t_pred).eval()
print('dice coefficient = {}'.format(a))
print('jaccard distance = {}'.format(b))
print('foreground recall = {}'.format(c))
sess.close()

# unit test
assert a == 1., 'WRONG DICE COEFFICIENT'
assert b == 0., 'WRONG JACCARD DISTANCE'
print('PASS!')