import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      
      # If margin is greater than zero, the class contributes to the loss,
      # and the gradient is computed
      if margin > 0:
        loss += margin
        dW[:,y[i]] -= X[i,:]
        dW[:,j] += X[i,:]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  
  # Normalize data gradient by number of training examples
  dW /= num_train
  
  # Add regularization part to the gradient
  dW += 2 * reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  # Done!

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  num_train = X.shape[0]
  delta = 1.0

  # Compute for the scores
  scores = X.dot(W)

  # Record the score of the example's correct class
  correct_class_score = scores[np.arange(num_train), y]

  # Compute for the margin by getting the max between 0 and the computed expression
  margins = np.maximum(0, scores - correct_class_score[:,np.newaxis] + delta)
  margins[np.arange(num_train), y] = 0

  # Add all the losses together
  loss = np.sum(margins)

  # Divide the loss all over the number of training examples
  loss /= num_train

  # Regularize
  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # pass
  # This mask can flag the examples in which their margin is greater than 0
  X_mask = np.zeros(margins.shape)
  X_mask[margins > 0] = 1

  # As usual, we count the number of these examples where margin > 0
  count = np.sum(X_mask,axis=1)
  X_mask[np.arange(num_train),y] = -count

  dW = X.T.dot(X_mask)

  # Divide the gradient all over the number of training examples
  dW /= num_train

  # Regularize
  dW += 2 * reg * W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
