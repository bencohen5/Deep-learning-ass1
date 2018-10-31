import loglinear as ll
import random
import numpy as np
from utils import TRAIN, DEV, vocab, L2I, F2I
import grad_check
STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


def feats_to_vec(features):
    vec = []
    for c in features:
        l1 = ord(c[0]) - 32
        l2 = ord(c[1]) - 32
        num = l1 * 96 + l2
        vec.append(num / 10000)
    for c in range(142 - len(features)):
        vec.append(0)
    return np.array(vec)


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions
        x = feats_to_vec(features)
        y = L2I[label]
        y_hat = ll.predict(x, params)
        if y - y_hat == 0:
            good += 1
        else:
            bad += 1
        pass
    return good / (good + bad)


def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.
    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """
    for I in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            gw, gb = grads
            cum_loss += loss
            W, b = params
            W -= learning_rate * gw
            b -= learning_rate * gb
            params = W, b
            # update the parameters according to the gradients
            # and the learning rate.

        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print
        I, train_loss, train_accuracy, dev_accuracy
    return params


if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    params = ll.create_classifier(142, 6)
    trained_params = train_classifier(TRAIN, DEV, 200, 0.001, params)
