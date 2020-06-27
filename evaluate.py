from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
import cv2
import sys
sys.path.append("..")

from attack.fast_gradient_method import fast_gradient_method
from attack.projected_gradient_descent import projected_gradient_descent

BASE_DIR = ''
# Original training dataset
TRAIN_DATA_DIR = ''
TRAIN_LABEL_DIR = ''
# Augmented dataset
AUG_DATA_DIR_1 = ''
AUG_DATA_DIR_2 = ''
AUG_DATA_DIR_3 = ''
AUG_FEATURES = ''

# Clean evaluation dataset
EVAL_DATA_DIR = ''
EVAL_LABEL_DIR = ''
EVAL_FEATURES = ''

# Adversarial perturbed evaluation dataset
ATTACK_DATA_DIR = ''
# Target attack label(used to calculate attack success rate)
ATTACK_LABEL_DIR = ''
# Features(hidden representations)
ATTACK_FEATURES = ''

BUFFER_SIZE = 10000
IMG_SIZE = 224
RESIZE_SIZE = 256
BATCH_SIZE = 128
K = 2048
EPOCHS = 100
EARLY_STOP = 5
LEARNING_RATE = 1e-5


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
        
def _training_data_generator(image_path, label):
    # Read image
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img.set_shape([None, None, 3])
        
    # Resize
    img = tf.image.resize(img, size=[RESIZE_SIZE, RESIZE_SIZE])
    img = tf.image.random_crop(img, size=(IMG_SIZE, IMG_SIZE, 3))

    # Augmentation
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.4)
    img = tf.image.random_contrast(img, lower=0.6, upper=1.4)
    img = tf.image.random_saturation(img, lower=0.6, upper=1.4)
    img = scale19(img)
    return img, label

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

def training_dataset_generator(img_path, label, data_generator, batch_size):
    assert img_path.shape[0] == label.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label))
    dataset = dataset.map(data_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def testing_dataset_generator(img_path, label, data_generator, batch_size):
    assert img_path.shape[0] == label.shape[0]
    dataset = tf.data.Dataset.from_tensor_slices((img_path, label))
    dataset = dataset.map(data_generator, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def feature_dataset_generator(feature, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices(feature)
    dataset = dataset.batch(batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
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

class RMC():
    def __init__(self, adapt_path=None, adapt_label=None, eval_path=None, 
                 eval_label=None, target_label=None, feature_ds=None):
        self.adapt_path = adapt_path
        self.adapt_label = adapt_label
        self.eval_path = eval_path
        self.eval_label = eval_label
        self.target_label = target_label
        self.feature_ds = feature_ds
        self.buffer_size = BUFFER_SIZE
        self.image_size = IMG_SIZE
        self.resize_size = RESIZE_SIZE
        self.batch_size = BATCH_SIZE
        self.k = K
        self.epochs = EPOCHS
        self.early_stop = EARLY_STOP
        self.learning_rate = LEARNING_RATE
        
        # Random sample
        self.idx = np.random.randint(len(eval_path), size=10000)
        
        # Load pretrained model
        self.resnet152v2 = tf.keras.applications.ResNet152V2(include_top=True,
                                                             weights='imagenet',
                                                             input_shape=(self.image_size, self.image_size, 3), 
                                                             classes=1000)
        
        # Evaluate the performance of RMC
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        self.robustness = tf.keras.metrics.SparseCategoricalAccuracy(name='robustness')
        self.success_rate = tf.keras.metrics.SparseCategoricalAccuracy(name='success_rate')
        
        # Evaluate the performance of our baseline, DeepNN and WebNN
        self.deepnn_robustness = 0
        self.deepnn_success_rate = 0
        self.webnn_robustness = 0
        self.webnn_success_rate = 0
        
    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            predictions = self.resnet152v2(images, training=False)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.resnet152v2.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.resnet152v2.trainable_variables))

        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        predictions = self.resnet152v2(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss.update_state(t_loss)
        self.test_accuracy.update_state(labels, predictions)

    @tf.function
    def adapt_step(self, images, labels, target_labels):
        predictions = self.resnet152v2(images, training=False)
        self.robustness.update_state(labels, predictions)
        self.success_rate.update_state(target_labels, predictions)
        
    def evaluate(self):
        for example_count, i in enumerate(self.idx):
            # Load test data
            adv_img = np.load(self.eval_path[i].replace('val_set', ATTACK_DATA_DIR).replace('JPEG', 'npy'))
            adv_img = tf.convert_to_tensor(adv_img)
            adv_img = adv_img[tf.newaxis, :]
            adv_img = tf.keras.applications.resnet_v2.preprocess_input(adv_img*255)
            label = tf.convert_to_tensor(self.eval_label[i])
            t_label = tf.convert_to_tensor(self.target_label[i])

            # Extract feature(precomputed), see more details in "/prepare/extract_feature.py"
            adv_repre = np.load(self.eval_path[i].replace('val_set', ATTACK_FEATURES).replace('JPEG', 'npy'))
            adv_repre = tf.convert_to_tensor(adv_repre)

            # Find KNN based on Euclidean distance, where K=4096
            l2_norm_list = []
            for feature in self.feature_ds:
                l2_norm = tf.norm(adv_repre - feature, ord='euclidean', axis=1)
                l2_norm_list.append(l2_norm.numpy())

            # Find KNN based on WebNN and DeepNN defense mechanisms
            l2_norm = np.hstack(l2_norm_list)
            knn_top20 = l2_norm.argsort()[:int(K*0.2)]
            knn_idx = l2_norm.argsort()[int(K*0.2):K]
            webnn_idx = l2_norm.argsort()[:K]
            deepnn_idx = l2_norm[3*int(len(self.adapt_label)/4):].argsort()[:K]
            
            # Local adaptation
            epoch = self.adapt(knn_top20, knn_idx, webnn_idx, deepnn_idx)
            
            # Evaluate DeepNN and WebNN
            webnn_count = np.bincount(self.adapt_label[webnn_idx])
            webnn_label = webnn_count.argmax()
            deepnn_count = np.bincount(self.adapt_label[deepnn_idx])
            deepnn_label = deepnn_count.argmax()

            if deepnn_label == label.numpy():
                self.deepnn_robustness += 1
            if deepnn_label == t_label.numpy():
                self.deepnn_success_rate += 1
            if webnn_label == label.numpy():
                self.webnn_robustness += 1
            if webnn_label == t_label.numpy():
                self.webnn_success_rate += 1

            # Evaluate RMC
            self.adapt_step(adv_img, label, t_label)
            
            print("Example {:d}, Step: {:d}".format(example_count+1, epoch+1))
            print("RMC: R: {:.2f}, SR: {:.2f} | WebNN: R: {:.2f}, SR: {:.2f} | DeepNN: R: {:.2f}, SR: {:.2f}".format(self.robustness.result()*100,
                                                                                                                     self.success_rate.result()*100,
                                                                                                                     (self.webnn_robustness/(example_count+1))*100,
                                                                                                                     (self.webnn_success_rate/(example_count+1))*100,
                                                                                                                     (self.deepnn_robustness/(example_count+1))*100,
                                                                                                                     (self.deepnn_success_rate/(example_count+1))*100))

#             with open('resnet152_adapt_info_16_conv5_1(100_v1).txt', 'a') as file:
#                 file.write("Example {:d}, Step: {:d} | ".format(example_count+1, epoch+1))
#                 file.write("RMC: R: {:.2f}, SR: {:.2f} | WebNN: R: {:.2f}, SR: {:.2f} | DeepNN: R: {:.2f}, SR: {:.2f}\n".format(self.robustness.result()*100,
#                                                                                                                      self.success_rate.result()*100,
#                                                                                                                      (self.webnn_robustness/(example_count+1))*100,
#                                                                                                                      (self.webnn_success_rate/(example_count+1))*100,
#                                                                                                                      (self.deepnn_robustness/(example_count+1))*100,
#                                                                                                                      (self.deepnn_success_rate/(example_count+1))*100))
            # Calibration
            self.calibrate()
            
        robustness.reset_states()
        success_rate.reset_states()
        
    def adapt(self, knn_top20, knn_idx, webnn_idx, deepnn_idx):
        # Random split top 20% NN examples to form a validation set
        X_train, X_test, y_train, y_test = train_test_split(self.adapt_path[knn_top20], 
                                                            self.adapt_label[knn_top20], 
                                                            test_size=0.25)
        
        # Create data for local adaptation
        X_train = np.concatenate((X_train, self.adapt_path[knn_idx]))
        y_train = np.concatenate((y_train, self.adapt_label[knn_idx]))
        BUFFER_SIZE = len(X_train)

        train_ds = training_dataset_generator(X_train, y_train, _testing_data_generator, BATCH_SIZE)
        eval_ds = testing_dataset_generator(X_test, y_test, _testing_data_generator, BATCH_SIZE)

        # Adapting with early stop
        min_loss = np.inf
        count = 0

        for epoch in range(EPOCHS):
            for images, labels in train_ds:
                images = tf.keras.applications.resnet_v2.preprocess_input(images*255)
                self.train_step(images, labels)

            for test_images, test_labels in eval_ds:
                test_images = tf.keras.applications.resnet_v2.preprocess_input(test_images*255)
                self.test_step(test_images, test_labels)

            # Record minimum loss
            if self.test_loss.result().numpy() < min_loss:
                min_loss = self.test_loss.result().numpy()
                count = 0
            else:
                count += 1

            template = 'Epoch {:0}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print (template.format(epoch+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            # Early stop if val loss does not decrease
            if count >= EARLY_STOP:
                print("Early stop at {:d} epoch.".format(epoch+1))
                break
                
        return epoch
        
    def calibrate(self):
        # Random sample K examples from augmented dataset
        random_idx = np.random.randint(len(self.adapt_path), size=K)
        X_train, X_test, y_train, y_test = train_test_split(self.adapt_path[random_idx], 
                                                            self.adapt_label[random_idx], 
                                                            test_size=0.125)
        BUFFER_SIZE = len(X_train)

        train_ds = training_dataset_generator(X_train, y_train, _training_data_generator, BATCH_SIZE)
        eval_ds = testing_dataset_generator(X_test, y_test, _testing_data_generator, BATCH_SIZE)

        # Calibrating with early stop
        min_loss = np.inf
        count = 0

        for epoch in range(EPOCHS):
            for images, labels in train_ds:
                images = tf.keras.applications.resnet_v2.preprocess_input(images*255)
                self.train_step(images, labels)

            for test_images, test_labels in eval_ds:
                test_images = tf.keras.applications.resnet_v2.preprocess_input(test_images*255)
                self.test_step(test_images, test_labels)

            # Record minimum loss
            if self.test_loss.result().numpy() < min_loss:
                min_loss = self.test_loss.result().numpy()
                count = 0
            else:
                count += 1

            template = 'Epoch {:0}, Loss: {:.2f}, Accuracy: {:.2f}, Test Loss: {:.2f}, Test Accuracy: {:.2f}'
            print (template.format(epoch+1,
                                   self.train_loss.result(),
                                   self.train_accuracy.result()*100,
                                   self.test_loss.result(),
                                   self.test_accuracy.result()*100))

            # Reset the metrics for the next epoch
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            # Early stop if val loss does not decrease
            if count >= EARLY_STOP:
                print("Early stop at {:d} epoch.".format(epoch+1))
                break
                
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
    feature_list = np.load(AUG_FEATURES)
    feature_ds = feature_dataset_generator(feature_list, 2048)
    
    rmc = RMC(aug_path, aug_label, eval_path, eval_label, target_label, feature_ds)
    rmc.evaluate()
    
if __name__ == '__main__':
    main()