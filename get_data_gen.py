"""
prepare data loader for training (excluding test set)
"""
import os
from data import trainGenerator


DATA_AUG_ARGS = dict(
        rotation_range = 45,
        horizontal_flip = True,
        vertical_flip = True,
        width_shift_range = 0.05,
        height_shift_range = 0.05,
        shear_range = 10.,
        zoom_range = 0.1,
        fill_mode = 'nearest'
        )


def get_data_gen(batch_size, train_path, aug_dict, target_size):
    """
    return data generator together with number of samples in the folder

    input:
        batch_size -- int, batch size
        train_path -- str, dir to "image", "label" folder
        aug_dict -- dict, dict of augmentation argument
        target_size -- tup, image shape for model input
    output:
        data_gen -- data generator
        data_n -- number of samples
    """
    data_gen = trainGenerator(batch_size = batch_size,
                              train_path = train_path,
                              image_folder = 'image',
                              mask_folder = 'label',
                              aug_dict = aug_dict,
                              target_size = target_size)
    data_n = len([i for i in os.listdir(os.path.join(train_path, 'image')) if i.endswith('.jpg')])
    return data_gen, data_n
