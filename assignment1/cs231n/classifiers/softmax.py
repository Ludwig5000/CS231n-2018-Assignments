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
  # regularization!                                                           #
  #############################################################################
  # pass
  num_classes = W.shape[1]
  num_train = X.shape[0]
  scores = X.dot(W)
  # Shift scores to avoid numerical instability
  scores -= np.max(scores)
  
  # Loop over images and classes
  for i in range(num_train):
    sum_i = np.sum(np.exp(scores[i,:]))
    p_ik = lambda k: np.exp(scores[i,k]) / sum_i
    
    # Compute loss
    loss += -np.log(p_ik(y[i]))
    
    # Compute gradient
    for k in range(num_classes):
      dW[:, k] += (p_ik(k) - (k == y[i])) * X[i,:]
      
  # Normalize and add regularization term
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
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
  # pass
  num_train = X.shape[0]
  scores = X.dot(W)
  # Shift scores to avoid numerical instability
  scores -= np.max(scores)
  
  sum_scores = np.sum(np.exp(scores), axis=1, keepdims=True)
  p = np.exp(scores)/sum_scores
  
  # Compute loss
  loss = np.sum(-np.log(p[np.arange(num_train), y]))

  # Compute gradient
  ind = np.zeros_like(p)
  ind[np.arange(num_train), y] = 1
  dW = X.T.dot(p - ind)
  
  # Normalize and add regularization term
  loss /= num_train
  loss += reg * np.sum(W * W)
  
  dW /= num_train
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

