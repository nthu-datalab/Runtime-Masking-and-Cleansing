from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import os
import cv2
import sys
sys.path.append("..")

from attack.fast_gradient_method_preprocess import fast_gradient_method
from attack.projected_gradient_descent_preprocess import projected_gradient_descent

BASE_DIR = ''
TRAIN_DATA_DIR = ''
TRAIN_LABEL_DIR = ''
EVAL_DATA_DIR = ''
EVAL_LABEL_DIR = ''
ATTACK_DATA_DIR = ''
ATTACK_LABEL_DIR = ''

# +
# PGD attack(norm=np.inf)
EPSILON = 16/255
EPS_ITERS = 1/255
NB_ITERS = 100
    
BUFFER_SIZE = 10000
IMG_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 64
TARGET = True
# -


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

def _testing_data_generator(image_path, label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = scale19(img)
    return img, label

def _attacking_data_generator(image_path, label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img, label

def _target_data_generator(image_path, label, target_label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
    img = scale19(img)
    return img, label, target_label

def testing_dataset_generator(img_path, label, dataset_generator, batch_size):
    assert img_path.shape[0] == label.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label))
    dataset = dataset.map(dataset_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def target_dataset_generator(img_path, label, target_label, dataset_generator, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label, target_label))
    dataset = dataset.map(dataset_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

@tf.function
def attack_step(model, metric, images, labels):
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
    
    # Use random classes as targeted classes in targeted attack
    print("Preparing dataset...")
    if TARGET:
        target_label = np.random.randint(1000, size=len(eval_label))
        # Remove replicate label
        for overlap in np.where(eval_label == target_label)[0]:
            new_label = np.random.randint(1000, size=1)
            while new_label != eval_label[overlap]:
                target_label[overlap] = new_label
                new_label = np.random.randint(1000, size=1)
        np.save(ATTACK_LABEL_DIR, target_label)
        # Prepare dataset with Tensorflow Dataset API
        test_ds = target_dataset_generator(eval_path, eval_label, target_label, _target_data_generator, BATCH_SIZE)
    else:
        # Prepare dataset with Tensorflow Dataset API
        test_ds = testing_dataset_generator(eval_path, eval_label, _testing_data_generator, BATCH_SIZE)
        
    # Load pretrained ResNet-152v2
    print("Loading pretrained model...")
    resnet152v2 = ResNet152V2()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    attack_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='attack_accuracy')

    adv_image_list = []
    print("Generating adversarial examples...")
    # Targeted attack
    if TARGET:
        for images, labels, target_labels in test_ds:
            adv_images = projected_gradient_descent(model_fn=resnet152v2, 
                                                    model_preprocess=tf.keras.applications.resnet_v2.preprocess_input, 
                                                    x=images,
                                                    eps=EPSILON, 
                                                    eps_iter=EPS_ITERS, 
                                                    nb_iter=NB_ITERS, 
                                                    norm=np.inf,
                                                    clip_min=None, 
                                                    clip_max=None, 
                                                    y=target_labels, 
                                                    targeted=True,
                                                    rand_init=True, 
                                                    rand_minmax=0.3,
                                                    sanity_checks=True)
            adv_image_list.append(adv_images.numpy())

            # Evaluate robustness and attack success rate
            _adv_images = tf.keras.applications.resnet_v2.preprocess_input(adv_images*255)
            test_step(resnet152v2, test_accuracy, _adv_images, labels)
            attack_step(resnet152v2, attack_accuracy, _adv_images, target_labels)
            
    # Non-targeted attack
    else:
        for images, labels in test_ds:
            adv_images = projected_gradient_descent(model_fn=resnet152v2, 
                                                    model_preprocess=tf.keras.applications.resnet_v2.preprocess_input, 
                                                    x=images,
                                                    eps=EPSILON, 
                                                    eps_iter=EPS_ITERS, 
                                                    nb_iter=NB_ITERS, 
                                                    norm=np.inf,
                                                    clip_min=None, 
                                                    clip_max=None, 
                                                    y=labels, 
                                                    targeted=False,
                                                    rand_init=True, 
                                                    rand_minmax=0.3,
                                                    sanity_checks=True)
            adv_image_list.append(adv_images.numpy())

            # Evaluate robustness
            _adv_images = tf.keras.applications.resnet_v2.preprocess_input(adv_images*255)
            test_step(resnet152v2, test_accuracy, _adv_images, labels)
    
    print("Done!\nRobustness: {:.2f}%".format(test_accuracy.result()*100))
    print("Attack Success Rate: {:.2f}%".format(attack_accuracy.result()*100))
    test_accuracy.reset_states()
    attack_accuracy.reset_states()
    
    adv_image_list = np.asarray(adv_image_list)
    adv_images = np.vstack(adv_image_list)

    # Save adversarial examples in single numpy file
    np.save(ATTACK_DATA_DIR+'.npy', adv_images)

    # Save adversarial examples in seperate numpy file
    for adv_image, adv_path in zip(adv_images, eval_path):
        np.save(adv_path.replace('val_set', ATTACK_DATA_DIR).replace('JPEG', 'npy'), adv_image)

if __name__ == '__main__':
    main()
