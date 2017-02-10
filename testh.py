from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
import numpy

def data():
    dataset = numpy.loadtxt("0207.csv", delimiter=",")
    X_train = dataset[:,0:8]
    Y_train = dataset[:,8]
    dataset2 = numpy.loadtxt("0208.csv", delimiter=",")
    X_test = dataset2[:,0:8]
    Y_test = dataset2[:,8]
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    model = Sequential()
    model.add(Dense({{choice([15, 512, 1024])}},input_dim=8,init='uniform', activation='softplus'))
    model.add(Dropout({{uniform(0, 1)}}))
    model.add(Dense({{choice([256, 512, 1024])}}))
    model.add(Activation({{choice(['relu', 'sigmoid','softplus'])}}))
    model.add(Dropout({{uniform(0, 1)}}))
    
    model.add(Dense(1, init='uniform', activation='sigmoid'))

    model.compile(loss='mse', metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    model.fit(X_train, Y_train,
              batch_size={{choice([10, 50, 100])}},
              nb_epoch={{choice([1, 50])}},
              show_accuracy=True,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))

    model_json = best_model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    best_model.save_weights("model.h5")
    print("Saved model to disk")


