from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

dataset2 = numpy.loadtxt("0208.csv", delimiter=",")
Z = dataset2[:,0:8]
Q = dataset2[:,8]
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
# test data
loaded_model.compile(loss='mse', optimizer='Adamax', metrics=['accuracy'])
score = loaded_model.evaluate(Z, Q, verbose=0)
print "for test %s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100)

# prediction
probabilities = loaded_model.predict(Z)
predictions = [float(round(x)) for x in probabilities]
accuracy = numpy.mean(predictions == Q)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

