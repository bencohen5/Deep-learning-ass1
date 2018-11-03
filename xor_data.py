import mlp1 as model
import train_mlp1 as trainer
from utils import I2L

data = [(I2L[1], [0, 0]),
        (I2L[0], [0, 1]),
        (I2L[0], [1, 0]),
        (I2L[1], [1, 1])]

if __name__ == "__main__":
    params = model.create_classifier(2, 10, 2)
    trained_params = trainer.train_classifier(data, data, 200, 0.0765, params)

