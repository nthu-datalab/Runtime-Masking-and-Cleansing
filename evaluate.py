from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from model import RMC

BASE_DIR = ''
# Original training dataset
TRAIN_DATA_DIR = ''
TRAIN_LABEL_DIR = ''
# Augmented dataset
AUG_DATA_DIR_1 = ''
AUG_DATA_DIR_2 = ''
AUG_DATA_DIR_3 = ''
AUG_FEATURES_DIR = ''

# Clean evaluation dataset
EVAL_DATA_DIR = ''
EVAL_LABEL_DIR = ''
EVAL_FEATURES_DIR = ''

# Adversarial perturbed evaluation dataset
ATTACK_DATA_DIR = ''
# Target attack label(used to calculate attack success rate)
ATTACK_LABEL_DIR = ''
# Features(hidden representations)
ATTACK_FEATURES_DIR = ''

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

def get_aug_path():
    # Prepare dataset
    # Training data
    train_path = np.load(BASE_DIR + TRAIN_DATA_DIR)
    train_label = np.load(BASE_DIR + TRAIN_LABEL_DIR)
    assert len(train_path) == len(train_label)
    
    pgd_path_10 = []
    pgd_path_30 = []
    pgd_path_100 = []

    for path in train_path:
        pgd_path_10.append(path.replace('train_set', AUG_DATA_DIR_1).replace('JPEG', 'png'))
        pgd_path_30.append(path.replace('train_set', AUG_DATA_DIR_2).replace('JPEG', 'png'))
        pgd_path_100.append(path.replace('train_set', AUG_DATA_DIR_3).replace('JPEG', 'png'))

    pgd_path_10 = np.asarray(pgd_path_10)
    pgd_path_30 = np.asarray(pgd_path_30)
    pgd_path_100 = np.asarray(pgd_path_100)

    aug_path = np.concatenate([pgd_path_10, pgd_path_30], axis=0)
    aug_path = np.concatenate([aug_path, pgd_path_100], axis=0)
    aug_path = np.concatenate([aug_path, train_path], axis=0)

    aug_label = np.concatenate([train_label, train_label], axis=0)
    aug_label = np.concatenate([aug_label, train_label], axis=0)
    aug_label = np.concatenate([aug_label, train_label], axis=0)
    assert len(aug_path) == len(aug_label)
    return aug_path, aug_label

def feature_dataset_generator(feature, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(feature)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def main():
    # Augmented data
    aug_path, aug_label = get_aug_path()
    
    # Evaluation data
    eval_path = np.load(BASE_DIR + EVAL_DATA_DIR)
    eval_label = np.load(BASE_DIR + EVAL_LABEL_DIR)
    assert len(eval_path) == len(eval_label)

    # Targeted class used in targeted-attack
    target_label = np.load(BASE_DIR + ATTACK_LABEL_DIR)
    
    # Load precomputed features(hidden representation)to accelerate searching k-NN
    # See more details in "/prepare/extract_feature.py"
    feature_list = np.load(AUG_FEATURES_DIR)
    feature_ds = feature_dataset_generator(feature_list, 2048)
    
    rmc = RMC(adapt_path=aug_path, adapt_label=aug_label, eval_path=eval_path, 
              eval_label=eval_label, target_label=target_label, feature_ds=feature_ds, 
              attack_data_dir=ATTACK_DATA_DIR, attack_feature_dir=ATTACK_FEATURES_DIR)
    rmc.evaluate()

if __name__ == '__main__':
    main()
