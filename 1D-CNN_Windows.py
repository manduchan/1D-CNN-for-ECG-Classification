#!/usr/bin/env python
# coding: utf-8

#---Install Packages---
# !pip install keras

import numpy as np 
import pandas as pd 
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Embedding, MaxPooling1D
from keras.layers import Conv1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
import time
from keras import metrics

print('import done')



#---Dataset: Previously sectioned ECG recordings into 2-second windows---

df = pd.read_csv('./Desktop/ECG_window_df.csv')

print('Dataset before the split:')
print(df.shape)


#---Train and Test split manually (test with patient 233 and 234 ECG windows)---

train = df.iloc[0:36900] 
test = df.iloc[-6300:]

X_train = train[train.columns[3:723]] #voltages, train
X_test = test[test.columns[3:723]]    #voltages, test
Y_train = train.Results               #results, train
Y_test = test.Results                 #results, test


print('Train Shape - voltages, results:')
print(X_train.shape, Y_train.shape)
print('Test Shape - voltages, results:')
print(X_test.shape, Y_test.shape) 


#---Model---

batch = 16
epochs = 10
shape = np.size(X_train,1)


model = Sequential()
model.add(Dense(100, activation='relu', input_shape = (shape,1)))
model.add(Conv1D(100, 10, activation='relu'))
model.add(Conv1D(100, 10, activation='relu'))
model.add(MaxPooling1D(3))
model.add(Conv1D(160, 10, activation='relu'))
model.add(Conv1D(160, 10, activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid')) 
model.summary()
model.compile(loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])


X_train = np.expand_dims(X_train, 2)
X_test = np.expand_dims(X_test, 2)


model.fit(X_train,Y_train, batch_size = batch, epochs = epochs)
score = model.evaluate(X_test, Y_test, batch_size = batch)
score

y_pred = model.predict(X_test, batch_size = batch)
threshold = 0.5
y_pred1 = y_pred[:,0] < threshold
cm = confusion_matrix(Y_test, y_pred1)
cm


#---Saving Model---

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
                              

