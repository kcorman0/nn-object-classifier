import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Normalize the error based on the number of samples in the category

# Go through each folder in a directory and add the images to an array.
# The folder name becomes the label for its files
def load_data(root_directory):
    subdirs = [x[0] for x in os.walk(root_directory)]       
    images = []
    labels = []                                                                     
    for subdir in subdirs:
        label = subdir.split('/')[5:][0]
        print("Subdirectory:", label)
        for img in tqdm(os.listdir(subdir)):
            label = subdir.split('/')[5:][0]
            path = os.path.join(subdir, img)
            img = cv2.cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)

            if img is not None:
                img = cv2.cv2.resize(img, (img_size, img_size))
                images.append(np.array(img)) # TODO is np.array necessary?
                labels.append(label)
            else:
                print("Image not loaded")
    return images, labels

# Displays 9 images on a plt subplot
def display_images(images, labels, pred_labels=None):
    assert len(images) == len(labels) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=255)

        if pred_labels is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], pred_labels[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

directory = "/Users/kippc/Downloads/101_ObjectCategories/"
# Label variables
classes = next(os.walk(directory))[1]
num_classes = len(classes)
# Image variables
img_size = 50 # TODO temp value
img_shape = (img_size, img_size)
# Data
train_images, train_labels = load_data(directory) # TODO separate train/test

# Input (the flattened image)
x = tf.placeholder(tf.float32, shape=[None, img_size**2], name="x")
# Output (what the network thinks the image is)
y = tf.placeholder(tf.float32, shape=[None, 1], name="y") # TODO shape for this and y_true may cause problems
# Correct label from the dataset
y_true = tf.placeholder(tf.float32, shape=[None, 1], name="y_true")

# Layer 1 variables
filters = 16
filter_size = 5
pooling_stride = 2
pooling_kernel = 2
# Layer 2 variables
filters2 = 32
filter_size2 = 5
pooling_stride2 = 2
pooling_kernel2 = 2
# Fully connected layer variables
fc_size = 128

# Initialize layer 1
x = tf.reshape(x, shape=[-1, img_size, img_size, 1])
conv1 = tf.layers.conv2d(x, filters, filter_size, activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, pooling_stride, pooling_kernel)
# Initialize layer 2
conv2 = tf.layers.conv2d(conv1, filters2, filter_size2, activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, pooling_stride2, pooling_kernel2)
# Initialize fully connected layer
fc1 = tf.contrib.layers.flatten(conv2)
fc1 = tf.layers.dense(fc1, fc_size)

session = tf.Session()
# session.run(tf.global_variable_initializer())
# print(session.run(conv2))

session.close()

# Print first 9 images (for testing)
display_images(images=train_images[0:9], labels=train_labels[0:9])