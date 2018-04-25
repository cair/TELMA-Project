from keras.models import Sequential
from keras.layers import Dense
import numpy
from sklearn import preprocessing
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

sp02_normalizer = preprocessing.Normalizer()
pulse_normalizer = preprocessing.Normalizer()

def sp02_to_msg(value):
    value = int(value)
    if value < 5:
        return 1
    elif value > 6:
        return 2
    
def pulse_to_msg(value):
    value = int(value)
    if value > 10:
        return 1
    elif value > 15:
        return 2
    
def one_hot(label, labels):
    vector = numpy.zeros(len(labels))
    vector[labels.index(label)] = 1
    return vector

def decode_predictions(predictions):
    temp = []
    for prediction in predictions:
        _max = max(prediction)
        for i, x in enumerate(prediction):
            i = i + 1
            if x == _max:
                temp.append(i)
                continue
    
    return temp


with open("COPD.csv") as f:
    dataset = [line.strip().split(',')[3:] for line in f.readlines()][1:]
    
sp02_values = []
pulse_values = []

data_x, data_y = [],[]
for line in dataset:
    
    line = [float(x) if x != "NULL" else -1.0 for x in line]

    sp02 = line[0]
    pulse = line[1]

    sp02_values.append(sp02)
    pulse_values.append(pulse)
    
    label = line.pop(-1)
    data_y.append(one_hot(label, [1, 2, 3]))
    
    line.append(sp02_to_msg(sp02))
    line.append(pulse_to_msg(pulse))
    data_x.append(line[:-1])
    
sp02_normalized = preprocessing.normalize([sp02_values], norm="l2")
pulse_normalized = preprocessing.normalize([pulse_values], norm="l2")

for i in range(len(data_x)):
    temp = []
    temp.append(sp02_normalized[0][i])
    temp.append(pulse_normalized[0][i])

    for j in range(2,len(data_x[i])):
        for x in one_hot(data_x[i][j], [1, 2, 3]):
            temp.append(x)
    data_x[i] = temp
    
print(data_y)

# split into input (X) and output (Y) variables
X = numpy.array(data_x)
Y = numpy.array(data_y)

# create model
model = Sequential()
model.add(Dense(12, input_dim=len(X[0]), init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(3, init='uniform', activation='softmax'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=3,  verbose=2)
# calculate predictions
predictions= model.predict(X)
# round predictions
print(decode_predictions(predictions))