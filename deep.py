from keras.models import Sequential
from keras.layers import Dense
import numpy
dataset = numpy.loadtxt("0207.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]
dataset2 = numpy.loadtxt("0208.csv", delimiter=",")
Z = dataset2[:,0:8]
Q = dataset2[:,8]

model = Sequential()
model.add(Dense(15, input_dim=8,init='uniform', activation='softplus'))
model.add(Dense(1, init='uniform', activation='sigmoid'))


model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])


history = model.fit(X, Y, nb_epoch=50, batch_size=10)


loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

