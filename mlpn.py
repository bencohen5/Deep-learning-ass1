import numpy as np
from utils import softmax, tanh

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}

ACTIVATION = tanh


def classifier_output(x, params):
    """
    Return the output layer (class probabilities)
    of a log-linear classifier with given params on input x.
    """
    h = x
    for w, b in zip(params[::2], params[1::2]):
        x = np.dot(h, w) + b
        h = ACTIVATION(x)
    probs = softmax(x)
    return probs


def predict(x, params):
    return np.argmax(classifier_output(x, params))


def get_layers(x, params):
    """
    returns the layers in reversed order. after activation func the layer is considered a new layer.
    :param x: input vector
    :param params: parameters list
    :return: list of the layers except last one
    """
    layers = [x, x]
    for w, b in zip(params[:-2:2], params[1:-2:2]):
        x = np.dot(x, w) + b
        layers = [x] + layers
        x = ACTIVATION(x)
        layers = [x] + layers
    return layers


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

    rev_params = list(reversed(params))
    grads = []
    layers = get_layers(x, params)
    last_dim = params[-1].size
    y_vec = np.zeros(last_dim)
    y_hat = classifier_output(x, params)
    y_vec[y] = 1
    loss = -1 * np.log(y_hat[y])
    g = y_hat - y_vec

    for h, z, b, w in zip(layers[::2], layers[1::2], rev_params[::2], rev_params[1::2]):
        gb = g
        gw = np.dot(np.transpose([h]), [g])
        gh = np.dot(g, np.transpose(w))
        gz = np.multiply(gh, ACTIVATION(z, derivative=True))
        grads = [gb] + grads
        grads = [gw] + grads
        g = gz

    return loss, grads


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
    for d1, d2 in zip(dims, dims[1:]):
        w = np.random.randn(d1, d2)
        b = np.random.randn(d2)
        params.append(w)
        params.append(b)
    return params


if __name__ == "__main__":
    p = create_classifier([100, 300, 600, 5])
    lrs = get_layers(np.zeros(100), p)
    _, grds = loss_and_gradients(np.ones(100), 2, p)
    print("hi")
