
# Object Classifier: Urban vs. Nature

The goal of this project was to create and train a convolutional neural network to be able to classify images as being urban or nature using Tensorflow.

## Dataset

I created my own 24,000 image dataset by scraping the first 12,000 results for "urban" and "nature" on Unsplash—a popular copyright-free photography website. These images had a lot of noise and were extremely different, which made the network much more difficult to train. The 24,000 images were split into 80% training data, 10% validation data and 10% testing data.

![Urban](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/urban_example.png)
![Nature](https://raw.githubusercontent.com/kcorman0/nn-object-classifier/master/nature_example.png)

## Neural Network

This is a diagram of the structure of my final neural network. Increasing the initial resolution would've likely given better results, but my GPU couldn't handle it.
![](https://puu.sh/BeSZ4/07402d59d3.png)
The network was designed to run 5000 iterations (each iteration is a batch of 128 images) before stopping, but if the validation accuracy is greater than 94% and the cost is less than .4 it moves on to test data.
## Results

![](https://puu.sh/Bc58v/99b45eab2f.png)

Test results from the same run as the graph.
![](https://puu.sh/BeT2Y/37ddbc3f18.png)
Some runs there were way more incorrect nature images (like this one) and others there were way more incorrect urban images. It seemed to depend on the initial weights.
