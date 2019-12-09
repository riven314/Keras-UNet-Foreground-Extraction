"""
check that:
1. data augmentation input shape matches
2. check data augmentation can be disabled
"""
import os
import sys
PATH = os.path.join(os.getcwd(), '..')
sys.path.append(PATH)

import matplotlib.pyplot as plt
from get_data_gen import DATA_AUG_ARGS, generate_data

BATCH_SIZE = 1
TARGET_SIZE = (480, 640)
N = 3 # no. of batch to be checked
DATA_AUG_ARGS = None

TRAIN_PATH = os.path.join('..', '..', '..', 'data', 'ready', 'train')
VAL_PATH = os.path.join('..', '..', '..', 'data', 'ready', 'val')
TEST_PATH = os.path.join('..', '..', '..', 'data', 'ready', 'test')
assert all(list(map(os.path.isdir, [TRAIN_PATH, VAL_PATH, TEST_PATH]))), '[ERROR] wrong data path'


train_gen, train_n = generate_data(batch_size = BATCH_SIZE,
                          train_path = TRAIN_PATH,
                          aug_dict = DATA_AUG_ARGS,
                          target_size = TARGET_SIZE)

val_gen, val_n = generate_data(batch_size = BATCH_SIZE,
                        train_path = VAL_PATH,
                        aug_dict = DATA_AUG_ARGS,
                        target_size = TARGET_SIZE)

test_gen, test_n = generate_data(batch_size = BATCH_SIZE,
                         train_path = TEST_PATH,
                         aug_dict = DATA_AUG_ARGS,
                         target_size = TARGET_SIZE)


def generate_sample_grid(data_gen, n = 3):
    img_ls, mask_ls = [], []
    # generate n rounds of batch samples
    for i, batch in enumerate(data_gen):
        if i == n:
            break
        img, mask = batch
        assert img.shape == mask.shape, 'image shape not matches with mask'
        assert img.shape == (BATCH_SIZE,) + TARGET_SIZE + (1,), 'target input shape not matches with image'
        img_ls.append(img[0, :, :, 0])
        mask_ls.append(mask[0, :, :, 0])
    # plot them
    f, axes = plt.subplots(n, 2)
    for i, (img, mask) in enumerate(zip(img_ls, mask_ls)):
        axes[i, 0].imshow(img)
        axes[i, 1].imshow(mask)
    plt.show()

if __name__ == '__main__':
    generate_sample_grid(train_gen, n = N)
    generate_sample_grid(val_gen, n = N)
    generate_sample_grid(test_gen, n = N)