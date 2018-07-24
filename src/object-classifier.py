import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from datetime import datetime
from tqdm import tqdm

# Go through each folder in a directory and add the images to an array.
# The folder name becomes the label for its images
def load_data(root_directory):
    subdirs = [x[0] for x in os.walk(root_directory)]
    subdirs.remove(root_directory)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    i = 0

    for subdir in subdirs:
        label = subdir.split('/')[5:][0]
        print("Subdirectory:", label)
        for img in tqdm(os.listdir(subdir)):
            path = os.path.join(subdir,img)
            img = cv2.cv2.imread(path,cv2.cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                i += 1
                img = cv2.cv2.resize(img, (img_size, img_size))
                label = subdir.split('/')[5:][0]
                label = classes.index(label)
                if i % (1 / test_percent) == 0:
                    test_images.append(np.array(img))
                    test_labels.append(np.array(label))
                else:
                    train_images.append(np.array(img))
                    train_labels.append(np.array(label))
            else:
                print("Image not loaded")
    return train_images, train_labels, test_images, test_labels

# Displays 9 images on a plt subplot
def display_images(images, labels, pred_labels=None):
    assert len(images) == len(labels) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_shape), cmap="gray", vmin=0, vmax=255)

        if pred_labels is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], pred_labels[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# TODO stop reusing same images (?)
# Randomly selects the indicated number of images/labels for training
def next_batch(size, images, labels):
    n = np.arange(0, len(images))
    np.random.shuffle(n)
    n = n[0:size]
    image_batch = [images[i] for i in n]
    label_batch = [labels[i] for i in n]
    return np.asarray(image_batch), np.asarray(label_batch)

# Change the network's output (which is an int) to a string representing the category it predicted
def argmax_to_label(prediction):
    labels = []
    for pred in prediction:
        labels.append(classes[pred])
    return labels

# Creates a confusion matrix thats shows the accuracy of each category
# def confusion_matrix(confusion):

# Constants
# Data variables
directory = "/Users/kippc/Downloads/PetImages/"
test_percent = .1
# Label variables
classes = next(os.walk(directory))[1]
num_classes = len(classes)
# Image variables
img_size = 50 # TODO temp value
img_shape = (img_size, img_size)
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
dropout_rate = 0.25
is_training = True
# Training variables
train_batch_size = 64
learning_rate = 0.0001 # TODO temp value
training_iterations = 10000
# Testing variables
test_batch_size = 256

# Data
train_images, train_labels, test_images, test_labels = load_data(directory)

# Input (the greyscale image from the dataset)
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size], name="x")
# Correct label for the input as an int
y = tf.placeholder(tf.int64, shape=[None], name="y")
# Correct label
# y_label = argmax_to_label(y) TODO better way to do this
# print("YYY",y_label)

# Initialize layer 1
x_reshape = tf.reshape(x, shape=[-1, img_size, img_size, 1])
conv1 = tf.layers.conv2d(x_reshape, filters, filter_size, activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, pool_size=pooling_kernel, strides=pooling_stride)
# Initialize layer 2
conv2 = tf.layers.conv2d(conv1, filters2, filter_size2, activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, pool_size=pooling_kernel2, strides=pooling_stride2)
# Initialize fully connected layers
fc1 = tf.layers.flatten(conv2)
fc1 = tf.layers.dense(fc1, fc_size, activation=tf.nn.relu)
fc1 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training)
fc2 = tf.layers.dense(fc1, num_classes)
# TODO Normalize the error based on the number of samples in the category
# Cost function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=y)
cost = tf.reduce_mean(cross_entropy)
# Train by minimizing the cost function
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Network's label prediction
pred = tf.argmax(fc2, 1)
# pred_labels = argmax_to_label(pred) TODO do this later in the program
correct_pred = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Start the Tensorflow session - nothing has been done with tensors until this points besides intializing
sess = tf.Session()
sess.run(tf.global_variables_initializer())
start_time = datetime.now()

# Train the network (training_iterations) amount of times
accuracies = []
for i in range(training_iterations):
    batch_images, batch_labels = next_batch(train_batch_size, train_images, train_labels)
    feed = {x: batch_images, y: batch_labels}
    sess.run(train, feed_dict=feed)

    if i % 100 == 0 or i == training_iterations - 1:
        acc = sess.run(accuracy, feed_dict=feed)
        accuracies.append(acc)
        print("\ncost:",sess.run(cost, feed_dict=feed))
        print("acc: {0:.2f}%".format(acc * 100))
# Mean training accuracy
total = 0
for num in accuracies:
    total += num
mean_accuracy = total / len(accuracies)

# Run the test data and come up with final numbers
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
is_training = False
batch_images, batch_labels = next_batch(test_batch_size, test_images, test_labels)
feed = {x: batch_images, y: batch_labels}
print("\nCompleted {0} iterations in {1} seconds".format(training_iterations, total_time))
print("Mean training accuracy: {0:.2f}%".format(mean_accuracy * 100))
print("\ngiven:",sess.run(y, feed_dict=feed))
print("cost:",sess.run(cost, feed_dict=feed))
print("prediction:", sess.run(pred, feed_dict=feed))
print("acc: {0:.2f}%".format(sess.run(accuracy, feed_dict=feed) * 100))
confusion = tf.confusion_matrix(labels = y, predictions = pred, num_classes = num_classes)
np.set_printoptions(threshold=np.nan)
print(sess.run(confusion, feed_dict=feed))
# TODO show individual accuracy of each category

sess.close()

# Print first 9 images (for testing)
display_images(images=train_images[0:9], labels=train_labels[0:9])