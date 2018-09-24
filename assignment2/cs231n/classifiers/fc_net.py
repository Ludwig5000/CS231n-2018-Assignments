#from builtins import range
#from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # initialize model parameters
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # define hidden layer
        out_hidden, cache_hidden = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        
        # define output layer
        scores, cache_scores = affine_forward(out_hidden, self.params['W2'], self.params['b2'])
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # calculate loss
        loss, dscores = softmax_loss(scores, y)
        
        # add regularization term to loss
        loss += self.reg * 0.5 * (
            np.sum(self.params['W1'] * self.params['W1']) +
            np.sum(self.params['W2'] * self.params['W2'])
        )
        
        # backpropagate into layers
        dhidden, grads['W2'], grads['b2'] = affine_backward(dscores, cache_scores)
        dX, grads['W1'], grads['b1'] = affine_relu_backward(dhidden, cache_hidden)
        
        # add regularization term to weight gradients
        grads['W2'] += self.reg * self.params['W2']
        grads['W1'] += self.reg * self.params['W1']
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # at least one hidden dim is required!
        if self.num_layers == 1:
            raise ValueError('No hidden layers spezified')
        
        # append list of layers by input_dim and num_classes
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        # loop over layer_dims
        for l in range(1, len(layer_dims)):
            W_key = 'W' + str(l)
            b_key = 'b' + str(l)
            self.params[W_key] = weight_scale * np.random.randn(layer_dims[l-1], layer_dims[l])
            self.params[b_key] = np.zeros(layer_dims[l])
        
        # initialize parameters for batch- or layer normalization
        if (normalization == 'batchnorm') or (normalization == 'layernorm'):
            for l in range(1, len(hidden_dims) +1):
                gamma_key = 'gamma' + str(l)
                beta_key = 'beta' + str(l)
                self.params[gamma_key] = np.ones(hidden_dims[l-1])
                self.params[beta_key] = np.zeros(hidden_dims[l-1])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        out = {}
        cache = {}
        cache_do = {}

        # loop over layers
        for l in range(1, self.num_layers + 1):
            W_key = 'W' + str(l)
            b_key = 'b' + str(l)
            gamma_key = 'gamma' + str(l)
            beta_key = 'beta' + str(l)
            
            # forward first layer
            if l == 1:
                if self.normalization=='batchnorm':
                    out[l], cache[l] = affine_batchnorm_relu_forward(
                        X, self.params[W_key], self.params[b_key], self.bn_params[0], self.params[gamma_key], self.params[beta_key]
                    )
                elif self.normalization=='layernorm':
                    out[l], cache[l] = affine_layernorm_relu_forward(
                        X, self.params[W_key], self.params[b_key], self.bn_params[0], self.params[gamma_key], self.params[beta_key]
                    )
                else:
                    out[l], cache[l] = affine_relu_forward(X, self.params[W_key], self.params[b_key])
                if self.use_dropout:
                    out[l], cache_do[l] = dropout_forward(out[l], self.dropout_param)
            # forward output layer        
            elif l == self.num_layers:
                scores, cache_scores = affine_forward(out[l-1], self.params[W_key], self.params[b_key])
            # forward intermediate layers
            else:
                if self.normalization=='batchnorm':
                    out[l], cache[l] = affine_batchnorm_relu_forward(
                        out[l-1], self.params[W_key], self.params[b_key], self.bn_params[l-1], self.params[gamma_key], self.params[beta_key]
                    )
                elif self.normalization=='layernorm':
                    out[l], cache[l] = affine_layernorm_relu_forward(
                        out[l-1], self.params[W_key], self.params[b_key], self.bn_params[l-1], self.params[gamma_key], self.params[beta_key]
                    )
                else:
                    out[l], cache[l] = affine_relu_forward(out[l-1], self.params[W_key], self.params[b_key])
                if self.use_dropout:
                    out[l], cache_do[l] = dropout_forward(out[l], self.dropout_param)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        dacts = {}
        
        # calculate loss
        loss, dscores = softmax_loss(scores, y)
        
        # loop over layers
        for l in range(self.num_layers, 0, -1):
            W_key = 'W' + str(l)
            b_key = 'b' + str(l)
            gamma_key = 'gamma' + str(l)
            beta_key = 'beta' + str(l)
            
            # add regularization part to loss
            loss += self.reg * 0.5 * (np.sum(self.params[W_key] * self.params[W_key]))
            
            # backprop output layer
            if l == self.num_layers:
                dacts[l], grads[W_key], grads[b_key] = affine_backward(dscores, cache_scores)
                grads[W_key] += self.reg * self.params[W_key]
            # backprop hidden layers
            else:
                if self.use_dropout:
                    dacts[l+1] = dropout_backward(dacts[l+1], cache_do[l])
                if self.normalization=='batchnorm':
                    dacts[l], grads[W_key], grads[b_key], grads[gamma_key], grads[beta_key] = affine_batchnorm_relu_backward(dacts[l+1], cache[l])
                elif self.normalization=='layernorm':
                    dacts[l], grads[W_key], grads[b_key], grads[gamma_key], grads[beta_key] = affine_layernorm_relu_backward(dacts[l+1], cache[l])
                else:
                    dacts[l], grads[W_key], grads[b_key] = affine_relu_backward(dacts[l+1], cache[l])
                grads[W_key] += self.reg * self.params[W_key]
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

def affine_batchnorm_relu_forward(x, w, b, bn_param, gamma, beta):
    """
    Convenience layer that perorms an affine transform followed by batch normalization and by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - bn_param: Parameter for batch normalization
    - gamma: Scale parameter for batch normalization
    - beta: Shift parameter for batch normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    
    fc, fc_cache = affine_forward(x, w, b)
    bn, bn_cache = batchnorm_forward(fc, gamma, beta, bn_param)
    out, relu_cache = relu_forward(bn)
    cache = (fc_cache, bn_cache, relu_cache)
    return out, cache
 
def affine_batchnorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, bn_cache, relu_cache = cache
    dbn = relu_backward(dout, relu_cache)
    dfc, dgamma, dbeta = batchnorm_backward(dbn, bn_cache)
    dx, dw, db = affine_backward(dfc, fc_cache)
    return dx, dw, db, dgamma, dbeta

def affine_layernorm_relu_forward(x, w, b, ln_param, gamma, beta):
    """
    Convenience layer that perorms an affine transform followed by layer normalization and by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer
    - ln_param: Parameter for layer normalization
    - gamma: Scale parameter for layer normalization
    - beta: Shift parameter for layer normalization

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    
    fc, fc_cache = affine_forward(x, w, b)
    ln, ln_cache = layernorm_forward(fc, gamma, beta, ln_param)
    out, relu_cache = relu_forward(ln)
    cache = (fc_cache, ln_cache, relu_cache)
    return out, cache
 
def affine_layernorm_relu_backward(dout, cache):
    """
    Backward pass for the affine-batchnorm-relu convenience layer
    """
    fc_cache, ln_cache, relu_cache = cache
    dln = relu_backward(dout, relu_cache)
    dfc, dgamma, dbeta = layernorm_backward(dln, ln_cache)
    dx, dw, db = affine_backward(dfc, fc_cache)
    return dx, dw, db, dgamma, dbeta