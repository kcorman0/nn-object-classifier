import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Normalize the error based on the number of samples in the category

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
classes = next(os.walk(directory))[1]
num_classes = len(classes)
img_size = 50 # TODO temp value
img_shape = (img_size, img_size)
train_images, train_labels = load_data(directory) # TODO separate train/test

# Input (the flattened image)
x = tf.placeholder(tf.float32, shape=[None, img_size**2], name="x")
# Output (what the network thinks the image is)
y = tf.placeholder(tf.float32, shape=[None, 1], name="y") # TODO shape for this and y_true may cause problems
# Correct label from the dataset
y_true = tf.placeholder(tf.float32, shape=[None, 1], name="y_true")

display_images(images=train_images[0:9], labels=train_labels[0:9])