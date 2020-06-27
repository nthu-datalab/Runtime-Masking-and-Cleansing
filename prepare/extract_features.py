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

BUFFER_SIZE = 10000
IMG_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 64
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)


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

def testing_dataset_generator(img_path, label, dataset_generator, batch_size):
    assert img_path.shape[0] == label.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label))
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

# Extract features from pretrained ResNet-152v2
class ResNet152V2_extractor(tf.keras.Model):
    def __init__(self, layers, trainable=False):
        super(ResNet152V2_extractor, self).__init__()
        self.resnet152v2 =  resnet_layers(layers)
        self.resnet152v2.trainable = trainable
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        
    # return the feature map of required layer
    def call(self, inputs):
        outputs = self.resnet152v2(inputs)
        outputs = self.gap(outputs)
        return outputs

def resnet_layers(layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    resnet152v2 = tf.keras.applications.ResNet152V2(include_top=True,
                                                    weights='imagenet',
                                                    input_shape=IMG_SHAPE, 
                                                    classes=1000)
    output = [resnet152v2.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([resnet152v2.input], output)
    return model

@tf.function
def extract_step(model, images):
    outputs = model(images)
    return outputs

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
    
    print("Preparing dataset...")
    # If the image size is 224*224, use "_attacking_data_generator" as dataset_generator in dataset API
    # Else use "_testing_data_generator" to crop the image first
    eval_ds = testing_dataset_generator(eval_path, eval_label, _testing_data_generator, BATCH_SIZE)
    # eval_ds = testing_dataset_generator(eval_path, eval_label, _attacking_data_generator, BATCH_SIZE)
    
    print("Creating feature extractor...")
    # In this paper, we extract features from convolutional layer 5_1
    # It is possible to extract from different hidden layers
    resnet152v2_extractor = ResNet152V2_extractor(layers=['conv5_block1_preact_bn'])
    
    print("Extracting features...")
    # evaluation dataset
    eval_pgd_feature_list = list()
    for adv_images, _ in eval_ds:
        adv_images = tf.keras.applications.resnet_v2.preprocess_input(adv_images*255)
        outputs = extract_step(resnet152v2_extractor, adv_images)
        eval_pgd_feature_list.append(outputs.numpy())
    
    eval_pgd_feature_list = np.asarray(eval_pgd_feature_list)
    eval_pgd_feature_list = np.vstack(eval_pgd_feature_list)
    eval_pgd_feature_list.shape

    print("Saving file...")
    # Save features in single numpy file
    np.save(ATTACK_DATA_DIR+'.npy', eval_pgd_feature_list)
    
    # Save features in seperate numpy file
    for adv_repre, adv_path in zip(eval_pgd_feature_list, eval_path):
        np.save(adv_path.replace('val_set', ATTACK_DATA_DIR).replace('JPEG', 'npy'), adv_repre)

if __name__ == '__main__':
    main()
