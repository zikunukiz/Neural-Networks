# CSC321 - Neural Network and Machine Learning
Assignments designed by Professor Roger Grosse
http://www.cs.toronto.edu/~rgrosse/

# Assignment Overviews
## 1. Loss Functions and Backprop
This assignment is meant to get your feet wet with computing the gradients for a model using
backprop, and then translating your mathematical expressions into vectorized Python code. It’s
also meant to give you practice reasoning about the behavior of different loss functions.

## 2. Caption Generation
In this assignment, we will train a multimodal log bilinear language model. In particular, we will
deal with a dataset which contains data of two modalities, i.e., image and text. An instance of
the dataset consists of an image and several associated sentences. Each sentence is a so-called
caption of the image which describe its content. The overall goal of the neural language model is
to generate the caption given an image. Note that a caption (sentence) is generated word by word
conditioned on both the image and a fixed size context. The context of the word just means a
fixed-size contiguous sequence of words ahead of it.

## 3. Recurrent Neural Network Language Model
In thie project, you will work on extending min-char-rnn.py, the vanilla RNN language model
implementation we covered in tutorial. This was written by Andrej Karpathy. You will experiment
with the Shakespeare dataset, which is shakespeare.txt in the starter code.

## 4. Image Completion with Mixture of Bernoulli
In this assignment, we’ll implement a probabilistic model which we can apply to the task of image
completion. Basically, we observe the top half of an image of a handwritten digit, and we’d like to
predict what’s in the bottom half. 

