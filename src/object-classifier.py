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

    train_images, train_labels = [], []
    test_images, test_labels = [], []
    i = 0
    for subdir in subdirs:
        label = subdir.split('/')[5:][0]
        print("Subdirectory:", label)
        for img in tqdm(os.listdir(subdir)):
            path = os.path.join(subdir,img)
            img = cv2.cv2.imread(path,cv2.cv2.IMREAD_COLOR)
            
            if img is not None:
                i += 1
                img = cv2.cv2.resize(img, img_shape)
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

# Randomly selects the indicated number of images/labels for the next training iteration
def next_batch(size, images, labels):
    n = np.arange(0, len(images))
    np.random.shuffle(n)
    n = n[0:size]
    image_batch = [images[i] for i in n]
    label_batch = [labels[i] for i in n]
    return np.asarray(image_batch), np.asarray(label_batch)

# Displays 9 images on a plt subplot
def display_images(images, labels, pred_labels=None):
    assert len(images) == len(labels) == 9
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(img_size, img_size, 3))

        if pred_labels is None:
            xlabel = "True: {0}".format(labels[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(labels[i], pred_labels[i])
        ax.set_xlabel(xlabel, fontsize=8.0)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Display a graph of accuracy vs. # of iterations
def display_graphs(accuracies, costs):
    x = []
    for i in range(0, training_iterations + 1, 100):
        x.append(i)

    fig, (axes1, axes2) = plt.subplots(2, 1, sharex=True)
    axes1.plot(x, accuracies)
    axes1.set_ylabel("Accuracy")
    axes2.plot(x, costs)
    axes2.set_ylabel("Cost")
    axes2.set_xlabel("Iterations")
    plt.show()

# Saves the important constant values and the test results to a text file
def write_to_txt(mean_acc, last_acc, test_acc, test_cost, confusion_matrix):
    f = open(os.path.join(txt_directory, "past_data.txt"), "a")
    f.write("Img size: {0}\tLayer 1 (filter: {1} Size: {2} Stride: {3})\tLayer 2 (filter: {4} Size: {5} Stride: {6})\tFC nodes: {7}\n"
        .format(img_size, filters, filter_size, pooling_stride, filters2, filter2_size, pooling_stride2, fc_size))
    f.write("Learning rate: {0} Training iterations: {1}\n"
        .format(learning_rate, training_iterations))
    f.write("Mean acc: {0}\tLast acc: {1}\tTest acc: {2}\tTest cost: {3}\n{4}\n\n"
        .format(mean_acc, last_acc, test_acc, test_cost, confusion_matrix))
    f.close()

# Change the network's output (which is an int) to a string representing the category it predicted
def argmax_to_labels(predictions):
    labels = []
    for pred in predictions:
        labels.append(classes[pred])
    return labels

# Correctly formats the confusion matrix using data labels
def format_confusion(confusion):
    width = 10
    for i in classes:
        print("\t\t" + i, end='')
    print()
    for idx, row in enumerate(confusion):
        print(classes[idx], end='')
        for i in row:
            print("\t\t{0}".format(i), end='')
        print()

# Data variables
directory = "/Users/kipp/Downloads/NN_Dataset2/"
txt_directory = "/Users/kipp/Desktop/Deep Learning/Tensorflow/nn-object-classifier/"
test_percent = .2
# Label variables
classes = next(os.walk(directory))[1]
num_classes = len(classes)
# Image variables
img_size = 128
img_shape = (img_size, img_size)
# Layer 1 variables
filters = 64
filter_size = 3
pooling_stride = 2
pooling_kernel = 2
# Layer 2 variables
filters2 = 128
filter2_size = 3
pooling_stride2 = 2
pooling_kernel2 = 2
# Fully connected layer variables
fc_size = 2048
fc2_size = 2048
dropout_rate = 0.2
is_training = True
# Training variables
train_batch_size = 64
learning_rate = 0.000025 # TODO temp value
training_iterations = 5000
# Testing variables
test_batch_size = 128

# Data
train_images, train_labels, test_images, test_labels = load_data(directory)

# Input (the RGB values from the image in the dataset)
x = tf.placeholder(tf.float32, shape=[None, img_size, img_size, 3], name="x")
# Correct label for the input as an int
y = tf.placeholder(tf.int64, shape=[None], name="y")

# Initialize layer 1
x_reshape = tf.reshape(x, shape=[-1, img_size, img_size, 3])
conv1 = tf.layers.conv2d(x_reshape, filters, filter_size, activation=tf.nn.relu)
conv1 = tf.layers.max_pooling2d(conv1, pool_size=pooling_kernel, strides=pooling_stride)
# Initialize layer 2
conv2 = tf.layers.conv2d(conv1, filters2, filter2_size, activation=tf.nn.relu)
conv2 = tf.layers.max_pooling2d(conv2, pool_size=pooling_kernel2, strides=pooling_stride2)
# Initialize fully connected layers
fc1 = tf.layers.flatten(conv2)
fc1 = tf.layers.dense(fc1, fc_size, activation=tf.nn.relu)
fc1 = tf.layers.dropout(fc1, rate=dropout_rate, training=is_training)
fc2 = tf.layers.dense(fc1, fc2_size, activation=tf.nn.relu)
fc2 = tf.layers.dropout(fc2, rate=dropout_rate, training=is_training)
output = tf.layers.dense(fc2, num_classes)
# Cost function
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=y)
cost = tf.reduce_mean(cross_entropy)
# Train by minimizing the cost function
train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Network's label prediction
pred = tf.argmax(output, 1)
correct_pred = tf.equal(pred, y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Start the Tensorflow session - nothing has been done with tensors until this points besides intializing
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
start_time = datetime.now()

# Train the network (training_iterations) amount of times, printing the accuracy and cost every 100 iterations
accuracies, costs = [], []
for i in range(training_iterations):
    batch_images, batch_labels = next_batch(train_batch_size, train_images, train_labels)
    feed = {x: batch_images, y: batch_labels}
    sess.run(train, feed_dict=feed)

    if i % 100 == 0 or i == training_iterations - 1:
        acc = sess.run(accuracy, feed_dict=feed)
        tcost = sess.run(cost, feed_dict=feed)
        accuracies.append(acc)
        costs.append(tcost)
        # epoch = i * train_batch_size / len(train_images)
        print("\nIteration {0}".format(i))
        print("Cost: {0:.4f}".format(tcost))
        print("Acc: {0:.2f}%".format(acc * 100))
mean_accuracy = sum(accuracies) / len(accuracies)
end_time = datetime.now()
total_time = (end_time - start_time).total_seconds()
is_training = False
print("\nCompleted {0} iterations in {1:.2f} seconds".format(training_iterations, total_time))
print("Mean training accuracy: {0:.2f}%".format(mean_accuracy * 100))
display_graphs(accuracies, costs)

# Run the test data and come up with final numbers
test_len = len(test_images)
test_preds = np.zeros(shape=test_len, dtype=int)
accuracies, costs = [], []
i = 0
while i < test_len:
    # Since the assigned batch size might not divide perfectly into the number of test images
    end = min(i + test_batch_size, test_len)
    batch_images, batch_labels = test_images[i:end], test_labels[i:end]
    feed = {x: batch_images, y: batch_labels}
    test_preds[i:end] = sess.run(pred, feed_dict=feed)
    accuracies.append(sess.run(accuracy, feed_dict=feed))
    costs.append(sess.run(cost, feed_dict=feed))
    i = end
test_acc = sum(accuracies) / len(accuracies)
test_cost = sum(costs) / len(costs)

# Print the test results
confusion = tf.confusion_matrix(labels = test_labels, predictions = test_preds, num_classes = num_classes)
confusion = sess.run(confusion, feed_dict=feed)
np.set_printoptions(threshold=np.nan)
print("\nTest data\nCost: {0:.4f}".format(test_cost))
print("Accuracy: {0:.2f}%\n".format(test_acc * 100))
format_confusion(confusion)
# Write the results to a text file
write_to_txt(mean_accuracy, acc, test_acc, test_cost, confusion)
sess.close()
# Display all the incorrect images
incorrect_images, incorrect_labels, correct_labels = [], [], []
for i in range(test_len):
    if test_preds[i] != test_labels[i]:
        incorrect_images.append(test_images[i])
        incorrect_labels.append(test_preds[i])
        correct_labels.append(test_labels[i])
incorrect_labels = argmax_to_labels(incorrect_labels)
correct_labels = argmax_to_labels(correct_labels)

i = 0
while i < len(incorrect_images) - 9:
    display_images(images=incorrect_images[i:i+9], labels=correct_labels[i:i+9], pred_labels=incorrect_labels[i:i+9])
    i += 9
