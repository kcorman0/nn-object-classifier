import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm

# Normalize the error based on the number of samples in the category

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
                # one_hot = tf.one_hot(classes.index(label), num_classes)
                # one_hot = tf.Session().run(one_hot) # TODO better way to do this ?
                label = subdir.split('/')[5:][0]
                label = classes.index(label)
                if i % 10 == 0:
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
    test = prediction
    # test = tf.constant(4)
    print("PREDICTION WHERE IS THIS:",test)
    # prediction = tf.get_default_graph().as_graph_def().node[0].attr["value"].tensor.int_val[0]
    test = tf.Session().run(test)
    print("AFTER TJE SJOT",test)
    label = classes[test]
    return label

# Constants
directory = "/Users/kippc/Downloads/101_ObjectCategories/"
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
is_training = True # TODO change this variable when not training
# Training variables
batch_size = 64
learning_rate = 0.0001 # TODO temp value

# Data
train_images, train_labels, test_images, test_labels = load_data(directory)

# Input (the greyscale image from the dataset)
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size], name="x")
# Correct label for the input (one-hot encoded)
y = tf.placeholder(tf.float32, shape=[None], name="y")
# Correct label as an int
y_int = tf.argmax(y, 1)

# Initialize layer 1
x_reshape = tf.reshape(x, shape=[-1, img_size, img_size, 1])
conv1 = tf.layers.conv2d(x_reshape, filters, filter_size, activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, pooling_stride, pooling_kernel)
# Initialize layer 2
conv2 = tf.layers.conv2d(conv1, filters2, filter_size2, activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, pooling_stride2, pooling_kernel2)
# Initialize fully connected layer
fc1 = tf.layers.flatten(conv2)
fc1 = tf.layers.dense(fc1, fc_size)
fc1 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training)
# Cost function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = fc1, labels=y_int)
cost = tf.reduce_mean(cross_entropy)
# Train (TODO)
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Network's label prediction
pred = tf.argmax(fc1, 1)
# pred = tf.Session().run([pred], feed_dict={x: test_images})[0]
# pred_label = argmax_to_label(pred)

sess = tf.Session()
batch_images, batch_labels = next_batch(batch_size, train_images, train_labels)
sess.run(tf.global_variables_initializer())

# np.set_printoptions(threshold='nan')
print(batch_images)
print(batch_labels)
print("PREDICTION:",sess.run(y, feed_dict={x: batch_images, y: batch_labels}))
# print(y_int)

sess.close()

# Print first 9 images (for testing)
display_images(images=train_images[0:9], labels=train_labels[0:9])