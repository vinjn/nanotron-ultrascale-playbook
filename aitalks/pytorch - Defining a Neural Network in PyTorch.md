Defining a Neural Network in PyTorch
====================================

Created On: Apr 17, 2020 | Last Updated: Feb 06, 2024 | Last Verified: Nov 05, 2024

Deep learning uses artificial neural networks (models), which are computing systems that are composed of many layers of interconnected units. By passing data through these interconnected units, a neural network is able to learn how to approximate the computations required to transform inputs into outputs. In PyTorch, neural networks can be constructed using the `torch.nn` package.

Introduction
------------

PyTorch provides the elegantly designed modules and classes, including `torch.nn`, to help you create and train neural networks. An `nn.Module` contains layers, and a method `forward(input)` that returns the `output`.

In this recipe, we will use `torch.nn` to define a neural network intended for the [MNIST dataset](hhttps://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST).

Setup
-----

Before we begin, we need to install `torch` if it isn’t already available.

pip install torch

Steps
-----

1.  Import all necessary libraries for loading our data
    
2.  Define and initialize the neural network
    
3.  Specify how data will pass through your model
    
4.  \[Optional\] Pass data through your model to test
    

### 1\. Import necessary libraries for loading our data

For this recipe, we will use `torch` and its subsidiaries `torch.nn` and `torch.nn.functional`.

import torch
import torch.nn as nn
import torch.nn.functional as F

### 2\. Define and initialize the neural network

Our network will recognize images. We will use a process built into PyTorch called convolution. Convolution adds each element of an image to its local neighbors, weighted by a kernel, or a small matrix, that helps us extract certain features (like edge detection, sharpness, blurriness, etc.) from the input image.

There are two requirements for defining the `Net` class of your model. The first is writing an \_\_init\_\_ function that references `nn.Module`. This function is where you define the fully connected layers in your neural network.

Using convolution, we will define our model to take 1 input image channel, and output match our target of 10 labels representing numbers 0 through 9. This algorithm is yours to create, we will follow a standard MNIST algorithm.

class Net([nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")):
    def \_\_init\_\_(self):
      super(Net, self).\_\_init\_\_()

      \# First 2D convolutional layer, taking in 1 input channel (image),
      \# outputting 32 convolutional features, with a square kernel size of 3
      self.conv1 \= [nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")(1, 32, 3, 1)
      \# Second 2D convolutional layer, taking in the 32 input layers,
      \# outputting 64 convolutional features, with a square kernel size of 3
      self.conv2 \= [nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")(32, 64, 3, 1)

      \# Designed to ensure that adjacent pixels are either all 0s or all active
      \# with an input probability
      self.dropout1 \= [nn.Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d "torch.nn.Dropout2d")(0.25)
      self.dropout2 \= [nn.Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d "torch.nn.Dropout2d")(0.5)

      \# First fully connected layer
      self.fc1 \= [nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear")(9216, 128)
      \# Second fully connected layer that outputs our 10 labels
      self.fc2 \= [nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear")(128, 10)

my\_nn \= Net()
print(my\_nn)

We have finished defining our neural network, now we have to define how our data will pass through it.

### 3\. Specify how data will pass through your model

When you use PyTorch to build a model, you just have to define the `forward` function, that will pass the data into the computation graph (i.e. our neural network). This will represent our feed-forward algorithm.

You can use any of the Tensor operations in the `forward` function.

class Net([nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module "torch.nn.Module")):
    def \_\_init\_\_(self):
      super(Net, self).\_\_init\_\_()
      self.conv1 \= [nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")(1, 32, 3, 1)
      self.conv2 \= [nn.Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d "torch.nn.Conv2d")(32, 64, 3, 1)
      self.dropout1 \= [nn.Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d "torch.nn.Dropout2d")(0.25)
      self.dropout2 \= [nn.Dropout2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout2d.html#torch.nn.Dropout2d "torch.nn.Dropout2d")(0.5)
      self.fc1 \= [nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear")(9216, 128)
      self.fc2 \= [nn.Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear "torch.nn.Linear")(128, 10)

    \# x represents our data
    def forward(self, x):
      \# Pass data through conv1
      x \= self.conv1(x)
      \# Use the rectified-linear activation function over x
      x \= [F.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu "torch.nn.functional.relu")(x)

      x \= self.conv2(x)
      x \= [F.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu "torch.nn.functional.relu")(x)

      \# Run max pooling over x
      x \= [F.max\_pool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.max_pool2d.html#torch.nn.functional.max_pool2d "torch.nn.functional.max_pool2d")(x, 2)
      \# Pass data through dropout1
      x \= self.dropout1(x)
      \# Flatten x with start\_dim=1
      x \= [torch.flatten](https://docs.pytorch.org/docs/stable/generated/torch.flatten.html#torch.flatten "torch.flatten")(x, 1)
      \# Pass data through \`\`fc1\`\`
      x \= self.fc1(x)
      x \= [F.relu](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.relu.html#torch.nn.functional.relu "torch.nn.functional.relu")(x)
      x \= self.dropout2(x)
      x \= self.fc2(x)

      \# Apply softmax to x
      output \= [F.log\_softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.log_softmax.html#torch.nn.functional.log_softmax "torch.nn.functional.log_softmax")(x, dim\=1)
      return output

### 4\. \[Optional\] Pass data through your model to test

To ensure we receive our desired output, let’s test our model by passing some random data through it.

\# Equates to one random 28x28 image
random\_data \= [torch.rand](https://docs.pytorch.org/docs/stable/generated/torch.rand.html#torch.rand "torch.rand")((1, 1, 28, 28))

my\_nn \= Net()
result \= my\_nn(random\_data)
print (result)

Each number in this resulting tensor equates to the prediction of the label the random tensor is associated to.

Congratulations! You have successfully defined a neural network in PyTorch.

Learn More
----------

Take a look at these other recipes to continue your learning:

*   [What is a state\_dict in PyTorch](https://pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html)
    
*   [Saving and loading models for inference in PyTorch](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_models_for_inference.html)