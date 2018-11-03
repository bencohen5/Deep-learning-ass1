import numpy as np
import loglinear as ll

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
    W, b, U, b_tag = params
    x_tag = np.tanh(x.dot(W) + b)
    probs = ll.softmax(x_tag.dot(U) + b_tag)
    return probs


def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))


def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    W, b, U, b_tag = params
    y_hat = classifier_output(x, params)
    y_vec = np.zeros(y_hat.shape)
    y_vec[y] = 1
    x_t = np.transpose([x])
    Z1 = x.dot(W) + b
    h1 = np.tanh(Z1)
    y_hat = classifier_output(x, params)
    gU = np.transpose([h1]) * ((y_hat - y_vec))
    gb_tag = y_hat - y_vec
    U_gb = np.array(U.dot(gb_tag))
    gb = np.multiply(U_gb, 1 - np.power(np.tanh(Z1), 2))
    gW = gb * x_t
    loss = -1 * np.log(y_hat[y])
    return loss, [gW, gb, gU, gb_tag]


def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(in_dim, hid_dim)
    b = np.random.rand(hid_dim)
    U = np.random.randn(hid_dim, out_dim)
    b_tag = np.random.rand(out_dim)
    params = [W, b, U, b_tag]
    return params
