
# Image Classifier: Urban vs. Nature

The goal of this project was to create and train a convolutional neural network to be able to classify images as being urban or nature using Tensorflow.

## Dataset

I created my own 24,000 image dataset by scraping the first 12,000 results for "urban" and "nature" on Unsplash—a popular copyright-free photography website. These images had a lot of noise and were extremely different, which made the network much more difficult to train. The 24,000 images were split into 80% training data, 10% validation data and 10% testing data.
Actual size examples of images in the dataset (before being compressed to 128x128):

![Urban](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/images/urban_example.png)
![Nature](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/images/nature_example.png)

## Neural Network

This is a diagram of the structure of my final neural network. Increasing the initial resolution would've likely given better results, but my GPU couldn't handle it.
![](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/images/nn-diagram.png)

The network was designed to run 5000 iterations (each iteration is a batch of 128 images) before stopping, but if the validation accuracy is greater than 94% and the cost is less than .4 it moves on to test data.
## Results

Graphs generated by my program:

![](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/images/results-graph.png)

Test results from the same run as the graph:

![](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/images/results.png)

Some runs had way more incorrect nature images (like this one) and others had way more incorrect urban images. It seemed to depend on the initial weights.

The testing accuracy was usually around 90% on images it had never seen before. This number could definitely be improved with further testing or access to additional hardware, though it feels like getting an extremely high percentage on this dataset would be very challenging since I generated it myself using such broad tags. It would be hard to make a network that can classify a skyscraper and a close up image of graffiti on a wall as being the same.
