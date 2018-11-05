import numpy as np

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def softmax(x):
    """
    Compute the softmax vector.
    x: a n-dim vector (numpy array)
    returns: an n-dim vector (numpy array) of softmax values
    """
    e_x = np.exp(x - np.max(x))
    x = e_x / e_x.sum()
    # Your code should be fast, so use a vectorized implementation using numpy,
    # don't use any loops.
    # With a vectorized implementation, the code should be no more than 2 lines.
    #
    # For numeric stability, use the identify you proved in Ex 2 Q1.
    return x


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    layers = create_layers(x, params)
    probs = softmax(layers[-1])
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    # if len(params) ==0 :
    # TODO
    y_hat = classifier_output(x, params)
    y_vec = np.zeros(y_hat.shape)
    y_vec[y] = 1
    layers = create_layers(x, params)
    grad = []
    g = y_hat - y_vec
    grad.insert(0, np.transpose([layers[-2]]) * g)#w*g last layer before soft max
    grad.insert(0, g)
    for (l, w, b) in zip(layers[-2::], params[-2:0:-2], params[-1:1:-2]):
        g = np.multiply(g.dot(np.transpose(w)), 1 - np.power(np.tanh(l), 2))
        gW = np.transpose([l]) * g
        gB = g
        grad.insert(0, gB)
        grad.insert(0, gW)
    loss = -1 * np.log(y_hat[y])
    return loss, grad


def create_layers(x, params):
    layers = []
    layers.append(np.tanh(x.dot(params[0]) + params[1]))
    for w, b in zip(params[2:-2:2], params[3:-2:2]):
        layers.append(np.tanh(layers[-1].dot(w) + b))
    layers.append(layers[-1].dot(params[-2]) + params[-1])#last layer without tanh
    return layers


def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    for (d1, d2) in zip(dims, dims[1:]):
        w = np.zeros((d1, d2))
        b = np.zeros(d2)
        params.append(w)
        params.append(b)
    return params
