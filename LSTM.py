labels = [1, 3, 5]
l = 3
if l:
    labels.remove(l)

use_seed = True
import pandas as pd
from collections import Counter
import numpy as np
import time
if use_seed:
    np.random.seed(1111)
else:
    np.random.seed(int(time.time()))
import pickle
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score, auc
from sklearn.preprocessing import Normalizer
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers.advanced_activations import LeakyReLU, PReLU



# a function to convert values into one-hot encoded arrays
def onehot_encode_1(values, labels=[1, 3, 5]):
    temp = np.zeros_like(labels)
    temp[labels.index(values[0])] = 1
    return np.r_[temp, values[1:]]

def onehot_encode(values, labels=[1, 3, 5]):
    res = []
    for val in values:
        temp = np.zeros_like(labels)
        temp[labels.index(val)] = 1
        res.append(temp)
    return np.array(res)

def convert_to_labels(values, labels=[1, 3, 5]):
    res = []
    for value in values:
        value = list(value)
        i = value.index(max(value))
        res.append(labels[i])
        
    return res

# Listing the name of the columns to load
#labels = [1, 3, 5]
columns = ['Merida ID', 'Assess_date_No_Time', 'question_Triage', 'Pulse', 'SO2', 'SpO2ReferenceValue', 'PulseReferenceValue', 'manual_Triage']

# Then load the csv with pandas, using only the above columns
df = pd.read_csv(open('u4h.csv'), usecols=columns)

# This line reduces the dataset from x --> y, 
df.dropna(how='any', inplace=True)

# Remove rows where label is equal to l:
#l = 3


if l:
    df.drop(df[df['manual_Triage'] == l].index, inplace=True)
 #   labels.remove(l)


#Normalize the data
spo2_scaler = Normalizer()
pulse_scaler = Normalizer()
spo2_scaler.fit([df['SO2']])
pulse_scaler.fit([df['Pulse']])
 
df['SO2'] = spo2_scaler.transform([df['SO2']])[0]
df['SpO2ReferenceValue'] = spo2_scaler.transform([df['SpO2ReferenceValue']])[0]
df['Pulse'] = pulse_scaler.transform([df['Pulse']])[0]
df['PulseReferenceValue'] = pulse_scaler.transform([df['PulseReferenceValue']])[0]
 
# With this:
scaler = Normalizer()

# loop over the data and separate into input / output,
# and also training / testing.
# To do this, first have to separate the "patients" into
# their own lists. has to be done in case the users don't have an equal
# number of rows associated with them.

# First get a list of all unique IDs
unique_ids = df['Merida ID'].unique()
#then create a dictionary with new dataframes associated with
# the unique IDs
data_frames = {_id: df[:][df.loc[:, 'Merida ID'] == _id] for _id in unique_ids}

# Count how many instances for each label
c = Counter(df['manual_Triage'].values)

# Dictionary for recording experiment results
experiments = {l: [] for l in labels}

sigma = 1
   
copies = int(c[1.0] / c[5.0])

    
k = 5
for _ in range(k):
    original_dataframes = list(data_frames.values())
    np.random.shuffle(original_dataframes)

    #import pdb;pdb.set_trace()

    # specifying how many days you want the RNN to take into consideration.
    # here: 
    # [
    #    [t-2],
    #    [t-1],
    #    [t]
    # ]
    days = 5
    days_lookahead = 0
    test_split = int(len(original_dataframes) * 0.5)

    # Uncomment this if you want to test using only one patient
    #test_split = -1

    noise_func = np.vectorize(lambda mu: np.random.normal(mu, sigma))

    dataset = []
    columns = ['Pulse', 'SO2', 'SpO2ReferenceValue', 'PulseReferenceValue'] # overwriting old variable from above.
    for patient in original_dataframes[:test_split]:
        """
        for _ in range(copies):
            d = patient
            #d[["SO2", "Pulse"]] = d[["SO2", "Pulse"]].apply(lambda mu: random.gauss(mu, sigma))
            # First turn the data frame into a numpy 2-D array with the selected columns. 
            # One containing the input values, and one containing the wanted output values.
            d_x, d_y = d[columns].values, d['manual_Triage'].values
            #import pdb;pdb.set_trace()
            #UPDATE
            # d_y = np.array([1 if y==3 else y for y in d_y])

            # This loop creates a kind of 'window' that slide accross the data.
            # The size of the window is the same size as 'days'. So store the values
            # inside the window as input data, and then pick the value for output that
            # is just outside the window. 
            for i in range(days, len(d) - days_lookahead):
                dataset.append((d_x[i - days:i], d_y[i + days_lookahead]))
        """        
        d = patient

        d_x, d_y = d[columns].values, d['manual_Triage'].values
        question_triage = np.reshape(d['question_Triage'].values, (-1, 1))
        for i in range(days, len(d) - days_lookahead):
            if d_y[i + days_lookahead] == 5.0:
                for _ in range(copies):
                    temp = question_triage[i - days:i] 
                    x = np.round(noise_func(d_x[i - days:i]))
                    x = scaler.transform(x)

                    dataset.append((np.concatenate((temp, x), axis=1), d_y[i + days_lookahead]))

            temp = question_triage[i - days:i]
            dataset.append((np.concatenate((temp, d_x[i - days:i]), axis=1), d_y[i + days_lookahead]))

    np.random.shuffle(dataset)
    data_x, data_y = zip(*dataset)

    test_x, test_y = [], []
     # overwriting old variable from above.
    for d in original_dataframes[test_split:]:
    #d = list(data_frames.values())[-1]
        # First turn the data frame into a numpy 2-D array with the selected columns. 
        # One containing the input values, and one containing the wanted output values.
        d_x, d_y = d[columns].values, d['manual_Triage'].values
        question_triage = np.reshape(d['question_Triage'].values, (-1, 1))
        d_x = scaler.transform(d_x)
        d_x = np.concatenate((question_triage, d_x), axis=1)
        #UPDATE
        # d_y = np.array([1 if y==3 else y for y in d_y])

        # This loop creates a kind of 'window' that slide accross the data.
        # The size of the window is the same size as 'days'. So store the values
        # inside the window as input data, and then pick the value for output that
        # is just outside the window. 
        for i in range(days, len(d) - days_lookahead):
            test_x.append(d_x[i - days:i])
            test_y.append(d_y[i + days_lookahead])

    # Next apply onehot function on all values, turning them into onehot vectors 
    # 
    # What happens here is that get np.apply_along_axis to loop over all the rows in data_x/y
    # and then sends the row in to my onehot function. The function turns the values into onehot
    # vectors and returns them. For data_y, this is easy, but for data_x it's more complicated
    # because of the structure of the array. 
    data_x = [[np.apply_along_axis(onehot_encode_1, 0, t) for t in x] for x in data_x] 
    data_x = np.reshape(data_x, (len(data_x), days, -1)) 
    data_y = np.apply_along_axis(onehot_encode, 0, data_y, labels)
    test_x = [[np.apply_along_axis(onehot_encode_1, 0, t) for t in x] for x in test_x] 
    test_x = np.reshape(test_x, (len(test_x), days, -1)) 
    test_y = np.apply_along_axis(onehot_encode, 0, test_y, labels)


    # The data is now almost ready. Next is to split into testing and training
    #split_factor = 0.8 # Adjust this number to change training / testing ratio
    #s_f = int(split_factor * len(data_x))

    #train_x, train_y = data_x[:s_f], data_y[:s_f]
    #test_x, test_y = data_x[s_f:], data_y[s_f:]

    class_weights = {i: 1 / len ([x for x in data_y if x[i] == 1]) for i in range(len(data_y[0]))}
    
    #class_weights = {
     #   0: 1 / len([x for x in data_y if x[0] == 1]),
        #1: 1 / len([x for x in data_y if x[1] == 1]),
    #    2: 1 / len([x for x in data_y if x[2] == 1])
    #}
    
    training_factor = int(len(data_x) * 7)

    # Now data is ready and we can create the RNN
    model = Sequential()
    model.add(LSTM(64, input_shape=(data_x[0].shape)))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    #model.add(Dense(64, activation="relu"))
    #model.add(Dropout(0.5))
    #UPDATE, added hidden layer
    #for i in range(50):
    #    model.add(Dense(128, activation="relu"))
    #    model.add(Dropout(0.5))
    #END update
    model.add(Dense(len(labels), activation="softmax"))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(data_x[:training_factor], data_y[:training_factor], epochs=50, verbose=1, batch_size=10, shuffle=True, validation_data=(test_x, test_y), class_weight=class_weights)
    predictions = model.predict(test_x)
    pred_labels = convert_to_labels(predictions, labels)
    true_labels = convert_to_labels(test_y, labels)

    statistics = {l: 0 for l in labels}

    for x, y in zip(pred_labels, true_labels):
        if x == y:
            statistics[y] += 1

    for k in statistics.keys():
        statistics[k] /= len(true_labels)

    for k, v in statistics.items():
        experiments[k].append(v)


save_data = True
if save_data:
    # Save the entire model here:
    model.save("model.h5")
    
    with open("experiments.pkl", "wb") as f:
        pickle.dump(experiments, f)

    del history.model

    with open("history.pkl", "wb") as f:
        pickle.dump(history, f)

    with open("predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)

    with open("true_labels.pkl", "wb") as f:
        pickle.dump(test_y, f) 