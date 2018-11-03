from utils import feats_to_vec, I2L


def predict(test_data, model):
    """
    predicts to file test.pred
    :param test_data: dataset to predict on
    :param model: the nn model, a prediction function and params as tuple
    :return: None
    """
    pred_func, params = model
    with open('test.pred', 'w') as pred_file:
        for label, features in test_data:
            x = feats_to_vec(features)
            p = pred_func(x, params)
            label = I2L[p]
            pred_file.write(label + '\n')
