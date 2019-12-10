"""
key metrics to be used:
1. Jaccard distance loss (1 - IoU)
2. Dice similarity coefficient

QUESTIONS:
1. what is the difference between metrics and loss?
    - loss is evaluated at raw model output, metrics can be evaluated at final prediction (e.g. after argmax)
    - loss can be an approximate of ultimate metrics (for ease of back-propagation)
2. dice coef and jaccard loss is applied on soft output or hard output?
3. in jaccard loss, does the "smooth" parameter matters?

REFERENCE:
1. [paper] Raw G-Band Chromosome Image Segmentation Using U-Net Based Neural Network
2. [kaggle] UNet on lung (with dice coef): https://www.kaggle.com/toregil/a-lung-u-net-in-keras
3. [keras-extension] jaccard distance loss implementation: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
4. [stackoverflow] implement jaccard distance lost: https://stackoverflow.com/questions/49284455/keras-custom-function-implementing-jaccard
5. [paper] What is a good evaluation measure for semantic segmentation?: http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf
"""
import tensorflow as tf
import keras.backend as K


def fg_recall(y_true, y_pred):
    """
    soft metrics

    key metrics for preventing loss of chromosomes

    input:
        y_true: The ground truth tensor. (M, H, W, C)
        y_pred: The predicted tensor, (M, H, W, C)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    gt = K.sum(y_true_f)
    intersection = K.sum(y_true_f * y_pred_f)
    return intersection / gt


def dice_coef(y_true, y_pred):
    """
    soft metrics

    input:
        y_true: The ground truth tensor. (M, H, W, C)
        y_pred: The predicted tensor, (M, H, W, C)
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def jaccard_distance(y_true, y_pred, smooth = 100):
    """
    soft metrics

    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    input:
        y_true: The ground truth tensor. (M, H, W, C)
        y_pred: The predicted tensor, (M, H, W, C)
        smooth: smoothing factor for gradient 
        is_soft: y_pred is soft output or not (i.e. prob)
    # References
        - implementation from kera lib: https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/losses/jaccard.py
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f), axis=-1)
    sum_ = K.sum(K.abs(y_true_f) + K.abs(y_pred_f), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

