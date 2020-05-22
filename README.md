Introduction
============

The project concerns implementing and evaluating classification algorithms.The
classification of the input data to recognize a 28 X 28 grayscale handwritten digit
MNIST image and identify the digit from 0 to 9. This is first done using logistic
regression by training on the training input data set of 60,000 images and the
target label that the input features translates to. Then the trained model is tested
on a testing data of 10,000 images and the target labels generated for the testing
data is then compared with the labels which the model outputs. Then we will
implement the second model single hidden layer Neural Network and train it on
the same number of the input data and test data and tune its hyper parameters.
Then we will implement the third model Convolutional Neural Network model
which is available as a library implemented in the toolbox and train the model on
the given data.

DataSets
--------
We will use MNIST dataset 

Testing Dataset
________________
t10k-images.idx3-ubyte

t10k-labels.idx1-ubyte

Training Dataset
________________
train-images.idx3-ubyte

train-labels.idx1-ubyte

a large database and is commonly used for training and testing by tuning many machine learning algorithmic models. The dataset contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm.


Tasks
=====
We will extract the MNIST 60,000 training and 10,000 testing input and labels data
with the feature values that will be equal to the size of one input testing image.
We will tune the hyper-parameters for the logistic regression that is the mini batch
size of the stochastic gradient descent for weight update and the number of
iterations on the overall the data of training set.
We will then tune the hyper-parameters for the Neural Network which include the
number of hidden units and the number of times the weights are updated iterating
over the training set.

For both the models we will randomly pick the input data from the set to train our
model and update the error gradient

Logistic Regression Model
-------------------------
We will have 1 of K coding scheme where is equal to 10 as the number of digits
varies from 0 to 9.
𝑝(𝐶$│𝑥) = 𝑦$ (𝑥) = exp (𝑎$)/Σexp (𝑎$)
Which gives us the approximated calculation of target value. 𝑎$ is the activation
and is given by
𝑎$ = 𝑤$
2𝑥 + 𝑏$
The gradient of the error function is given by
∇67𝐸 𝑥 = 𝑦7 − 𝑡7 𝑥
Then we will choose learning rate whose initial value will be 1 and with each
iteration over the data. The learning rate will be decreased by the iteration rate.
We will update weights for the logistic regression model using learning rate and
the error gradient.

We will take a mini batch of size 100 and take 10 iterations over the data, Which
gives the error percent of 7.6 on training set and error percent of 7.9 on testing
set.

1. Training set Error Rate : 7.3

2. Testing set Error Rate : 7.9

3. Learning rate used for logistic regression : 1 which is decreased with each iteration.

4. Mini Batch Size : 100

5. Number of iterations : 10

Logistic Regression: Error Percent vs Number of iterations
![alt tag](https://github.com/exceptionhandle/HandWritten-Digit-Recognition/blob/master/logistic10.png)
Single Layer Neural Network
---------------------------

Layer 1 corresponds to input layer with bias 1 added

Layer2 corresponds to hidden layer with bias 1 added

Layer3 is the output layer giving the calculated output value.

We will use a neural network of single layer taking hidden units of 400

Minibatch size of 100 and the number of iterations = 3000

We will randomly decide initial weights ranging from 0 to 1.

We will use a fixed learning rate of 0.1 and a bias of 1 in input and the activation function output of hidden units.

The feed forward propragation is carried out as follows

The input for the hidden units is calculated by the hidden unit weights and the input at the input units. The inputs to the hidden units is then used to calculate activation value by using logistic sigmoid as the activation function.

To update the hidden weights and the output weights we do backpropagation
using the the calculated output target value and given target value. Then using the
delta difference between then to update the output weights .

The deactivation function translates the error in the output layer to the hidden layer and thereby updates the hidden weights using the the difference between the calculated output of the hidden weights and the backpropagated deactivation value.

1. Training set Error Rate : 1.9

2. Testing set Error Rate : 2.4

3. Learning Rate : 0.1

4. Number of iterations : 3000

Single Layer Neural Network : Error Rate vs Number of iterations

![alt tag](https://github.com/exceptionhandle/HandWritten-Digit-Recognition/blob/master/neural.png)

Convolutional Neural Network

In CNN, we transform the 2-D images into a 3-D space by convolving the image with 2 kernel filters for mapping the particular features from the image. After convolution we get k*(M-m+1)*(N-n+1) images.
where k = 2. We have images of size = 28 X 28 . So M = 28. We are using kernel size of m = 5X5. So m = 5, n = 5.

Convolving improves the feature by increasing the influence of the nearby pixels.

We multiply the convolution kernels with the input images and calculate the activation for that region.
Convolutional Neural Network
----------------------------
The figure above shows the image which is taken input of size 28 X 28 and convolved with the kernels

In convolution layer same filter is replicated and used over all locations of the image to extract the features of the image and all the the convolution kernels are trained with different learning gradient.

Feed Forward Propagation
------------------------
Each hidden neuron has a bias and 5X5 weights which is connected to the local receptive field. We will be using same weights for each of the 24 X 24 hidden layer neurons
Neural Activation Function
--------------------------
Convolutional network uses sigmoid function.

As we are using two hidden layers , we will have hidden layer output of 3 X 24 X 24.

Back-propagation
----------------
The backward pass for a convolution operation (for both the data and the weights) is also a convolution (but with spatially-flipped filters).

Convolution Parameters and Error
--------------------------------
I have used toolbox available from the site 
► https://github.com/rasmusbergpalm/DeepLearnToolbox

Used following hyper-Parameters and got the output error for training and testing.

Number of epochs of convolutional Neural network: 100

Learning rate : 1

Batch Size : 50

Training Error : 0.0067

Testing Error : 1.967
