use_seed = True
from collections import Counter
import pandas as pd
import numpy as np
if use_seed:
    np.random.seed(1337)
else:
    np.random.seed(int(time.time()))
 
from sklearn import svm
 
# Listing the name of the columns to load
labels = [1, 3, 5]
columns = ['Merida ID', 'Assess_date_No_Time', 'question_Triage', 'Pulse', 'SO2', 'SpO2ReferenceValue', 'PulseReferenceValue', 'manual_Triage']
 
# Then load the csv with pandas, using only the above columns
df = pd.read_csv(open('u4h.csv'), usecols=columns)
 
# This line reduces the dataset from x --> y,
df.dropna(how='any', inplace=True)
 
# Remove rows where label is equal to l:
l = 3
if l:
    df.drop(df[df['manual_Triage'] == l].index, inplace=True)
    labels.remove(l)
 
# First get a list of all unique IDs
unique_ids = df['Merida ID'].unique()
#then create a dictionary with new dataframes associated with
# the unique IDs
data_frames = {_id: df[:][df.loc[:, 'Merida ID'] == _id] for _id in unique_ids}
 
# Count how many instances for each label
c = Counter(df['manual_Triage'].values)
 
# Dictionary for recording experiment results
experiments = {l: [] for l in labels}
 
add_noise = True
 
if add_noise:
    sigma = 1
    try:
        copies = int(c[1.0] / c[5.0])
    except ZeroDivisionError:
        copies = .0
   
original_dataframes = list(data_frames.values())
np.random.shuffle(original_dataframes)
 
days = 5
days_lookahead = 0
test_split = int(len(original_dataframes) * 0.5)
 
# Uncomment this if you want to test using only one patient
#test_split = -1
 
noise_func = np.vectorize(lambda mu: np.random.normal(mu, sigma))
 
dataset = []
columns = ['Pulse', 'SO2', 'SpO2ReferenceValue', 'PulseReferenceValue'] # overwriting old variable from above.
for patient in original_dataframes[:test_split]:
    d = patient
   
    d_x, d_y = d[columns].values, d['manual_Triage'].values
    question_triage = np.reshape(d['question_Triage'].values, (-1, 1))
    for i in range(days, len(d) - days_lookahead):
        if d_y[i + days_lookahead] == 5.0 and add_noise:
            for _ in range(copies):
                temp = question_triage[i - days:i]
                x = noise_func(d_x[i - days:i])
 
                dataset.append((np.concatenate((temp, x), axis=1), d_y[i + days_lookahead]))
 
        temp = question_triage[i - days:i]
       
        dataset.append((np.concatenate((temp, d_x[i - days:i]), axis=1), d_y[i + days_lookahead]))
 
np.random.shuffle(dataset)
data_x, data_y = zip(*dataset)
 
test_x, test_y = [], []
columns = ['question_Triage', 'Pulse', 'SO2', 'SpO2ReferenceValue', 'PulseReferenceValue'] # overwriting old variable from above.
for d in original_dataframes[test_split:]:
    d_x, d_y = d[columns].values, d['manual_Triage'].values
 
    for i in range(days, len(d) - days_lookahead):
        test_x.append(d_x[i - days:i])
        test_y.append(d_y[i + days_lookahead])
 
data_x = np.array([x.reshape(-1) for x in data_x])
test_x = np.array([x.reshape(-1) for x in test_x])
 
classifier = svm.SVC()
classifier.fit(data_x, data_y)
print(classifier.score(test_x, test_y))
 
from sklearn.metrics import confusion_matrix
predictions = classifier.predict(test_x)
confusion_matrix(test_y, predictions)