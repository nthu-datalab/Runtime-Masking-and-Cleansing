# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import cv2
import os
import sys
sys.path.append("..")

from attack.fast_gradient_method_preprocess import fast_gradient_method
from attack.projected_gradient_descent_preprocess import projected_gradient_descent

BASE_DIR = ''
# Original training dataset
TRAIN_DATA_DIR = ''
TRAIN_LABEL_DIR = ''
# Augmented dataset
AUG_DATA_DIR = ''

# Clean evaluation dataset
EVAL_DATA_DIR = ''
EVAL_LABEL_DIR = ''

# Adversarial perturbed evaluation dataset
ATTACK_DATA_DIR = ''
# Target attack label(used to calculate attack success rate)
ATTACK_LABEL_DIR = ''

# +
# PGD attack(norm=np.inf)
EPSILON = 16/255
EPS_ITERS = 0.01
NB_ITERS = 10

BUFFER_SIZE = 10000
IMG_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 64
# -


# GPU settings
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

def _attacking_data_generator(image_path, label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    
    # Resize
    initial_width = tf.shape(img)[0]
    initial_height = tf.shape(img)[1]
    
    ratio = 0
    if(initial_width < initial_height):
        ratio = tf.cast(256 / initial_width, tf.float32)
        h = tf.cast(initial_height, tf.float32) * ratio
        img = tf.image.resize(img, (256, h), method="bicubic")
    else:
        ratio = tf.cast(256 / initial_height, tf.float32)
        w = tf.cast(initial_width, tf.float32) * ratio
        img = tf.image.resize(img, (w, 256), method="bicubic")
        
    img = tf.image.random_crop(img, size=(IMG_SIZE, IMG_SIZE, 3))
    return img, label

def _testing_data_generator(image_path, label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = scale19(img)
    return img, label

def testing_dataset_generator(img_path, label, _data_generator, batch_size):
    assert img_path.shape[0] == label.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label))
    dataset = dataset.map(_data_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def resize_image(image, shape=(224,224)):
    target_width = shape[0]
    target_height = shape[1]
    initial_width = tf.shape(image)[0]
    initial_height = tf.shape(image)[1]
    im = image
    ratio = 0
    if(initial_width < initial_height):
        ratio = tf.cast(256 / initial_width, tf.float32)
        h = tf.cast(initial_height, tf.float32) * ratio
        im = tf.image.resize(im, (256, h), method="bicubic")
    else:
        ratio = tf.cast(256 / initial_height, tf.float32)
        w = tf.cast(initial_width, tf.float32) * ratio
        im = tf.image.resize(im, (w, 256), method="bicubic")
    width = tf.shape(im)[0]
    height = tf.shape(im)[1]
    startx = width//2 - (target_width//2)
    starty = height//2 - (target_height//2)
    im = tf.image.crop_to_bounding_box(im, startx, starty, target_width, target_height)
    return im

def scale19(image):
    i = resize_image(image, (224,224))
    return (i)

def ResNet152V2():
    IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
    resnet152v2 = tf.keras.applications.ResNet152V2(include_top=True,
                                                    weights='imagenet',
                                                    input_shape=IMG_SHAPE, 
                                                    classes=1000)
    return resnet152v2

@tf.function
def test_step(model, metric, images, labels):
    predictions = model(images)
    metric.update_state(labels, predictions)

def main():
    # Prepare dataset
    # Training data
    train_path = np.load(BASE_DIR + TRAIN_DATA_DIR)
    train_label = np.load(BASE_DIR + TRAIN_LABEL_DIR)
    assert len(train_path) == len(train_label)

    # Evaluation data
    eval_path = np.load(BASE_DIR + EVAL_DATA_DIR)
    eval_label = np.load(BASE_DIR + EVAL_LABEL_DIR)
    assert len(eval_path) == len(eval_label)
    
    # Prepare dataset with Tensorflow Dataset API
    print("Preparing dataset...")
    train_ds = testing_dataset_generator(train_path, train_label, _attacking_data_generator, BATCH_SIZE)
    
    # Load pretrained ResNet-152v2
    print("Loading pretrained model...")
    resnet152v2 = ResNet152V2()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    
    print("Generating adversarial examples...")
    for images, labels in train_ds:
        adv_images = projected_gradient_descent(model_fn=resnet152v2, 
                                                model_preprocess=tf.keras.applications.resnet_v2.preprocess_input, 
                                                x=images,
                                                eps=EPSILON, 
                                                eps_iter=EPS_ITERS, 
                                                nb_iter=NB_ITERS, 
                                                norm=np.inf,
                                                clip_min=None, 
                                                clip_max=None, 
                                                y=None, 
                                                targeted=False,
                                                top_k=None,
                                                rand_init=True,
                                                rand_minmax=0.3,
                                                sanity_checks=True)

        # Save image as png file (less compression than jpg)
        for idx, (image, adv_image) in enumerate(zip(images, adv_images)):
            save_path = '/'.join(train_path[path_idx].split('/')[:-1]).replace('train_set', AUG_DATA_DIR) + '/'
            save_image = train_path[path_idx].split('/')[-1].replace('JPEG', 'png')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # save image with lower compression—bigger file size but faster decoding
            cv2.imwrite(save_path + save_image, 
                        cv2.cvtColor(adv_image.numpy()*255, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # Evaluate robustness
        _adv_images = tf.keras.applications.resnet_v2.preprocess_input(adv_images*255)
        test_step(resnet152v2, test_accuracy, _adv_images, labels)
    
    print("Done!\nTest Accuracy: {:.2f}%".format(test_accuracy.result()*100))
    test_accuracy.reset_states()

if __name__ == '__main__':
    main()
