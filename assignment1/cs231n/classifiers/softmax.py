import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                          #
  #############################################################################
  num_train = X.shape[0]
  dim = W.shape[0]
  num_classes = W.shape[1]

  for i in range(num_train):
      scores = X[i].reshape(1,-1) @  W
      softmax = np.exp(scores) / np.sum(np.exp(scores))
      softmax = softmax.reshape(-1)
      loss += -np.log(softmax[y[i]])
      label = np.zeros(num_classes)
      label[y[i]] = 1
      d_softmax = softmax - label
      dW += X[i].reshape(-1,1) @ d_softmax.reshape(1,-1)
  #normalization
  loss /= num_train
  dW /= num_train

  #regularization
  loss += reg * np.sum(W * W) 
  dW += 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]

  scores = X @ W
  softmax = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1,1)
  loss = np.sum(-np.log(softmax[range(num_train),y])) / num_train + reg * np.sum(W * W)
  label = np.zeros_like(softmax)
  label[range(num_train),y] = 1
  dW = X.T @ (softmax - label) / num_train + 2 * reg * W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

