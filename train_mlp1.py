import mlp1 as ll
import random
from utils import TRAIN, DEV, TEST, feats_to_vec, L2I
from predict_test import predict
import numpy as np

STUDENT = {'name': 'YOUR NAME',
           'ID': 'YOUR ID NUMBER'}


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
    for epoch in range(num_iterations):
        cum_loss = 0.0  # total loss in this iteration.
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features)  # convert features to a vector.
            y = L2I[label]  # convert the label to number if needed.
            loss, grads = ll.loss_and_gradients(x, y, params)
            gw, gb, gu, gb_tag = grads
            cum_loss += loss
            W, b, U, b_tag = params
            W -= learning_rate * gw
            b -= learning_rate * gb
            U -= learning_rate * gu
            b_tag -= learning_rate * gb_tag
            params = W, b, U, b_tag
            # update the parameters according to the gradients
            # and the learning rate.
        if epoch == 4:
            learning_rate = 0.001
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print(epoch, train_loss, train_accuracy, dev_accuracy)
    return params


if __name__ == '__main__':
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    params = ll.create_classifier(600, 1200, 6)
    trained_params = train_classifier(TRAIN, DEV, 10,  0.1, params)
    predict(TEST, (ll.predict, params))
