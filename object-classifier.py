import tensorflow as tf
import numpy as np
import cv2
import os
from tqdm import tqdm

# Normalize the error based on the number of samples in the category

def load_data(root_directory):
    subdirs = [x[0] for x in os.walk(root_directory)]                                                                            
    for subdir in subdirs:
        for img in tqdm(os.listdir(subdir)):
            label = subdir.split('/')[5:][0]
            path = os.path.join(subdir, img)
            img = cv2.cv2.imread(path, cv2.cv2.IMREAD_GRAYSCALE)
            images = []
            labels = []
            print("Subdirectory:", label)

            if img is not None:
                img = cv2.cv2.resize(img, (img_size, img_size))
                images.append(np.array(img)) # TODO is np.array necessary?
                labels.append(label)
            else:
                print("Image not loaded")
    return images, labels

# def display_images(images, labels):

directory = "/Users/kippc/Downloads/101_ObjectCategories/"
classes = next(os.walk(directory))[1]
num_classes = len(classes)
img_size = 50 # TODO temp value

train_images, train_labels = load_data(directory)

# display_images(images=train_images[0:9], labels=train_labels[0:9])