import os
import sys
import json
import time
import tensorflow as tf

#from tensorflow import keras # for building Neural Networks
#print('Tensorflow/Keras: %s' % keras.__version__) # print version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed

import pyrebase
import csv

set_seed(455)
np.random.seed(455)

print("KLASIFIKASI TANAMAN MENGGUNAKAN RNN")
print("===================================")
print("Konfigurasi firebase")

config = {
    'apiKey': "AIzaSyAJSE7jfLenK866bhQKVz0JTHpxk6P8hM8",
    'authDomain': "ta-ta-f5151.firebaseapp.com",
    'databaseURL': "https://ta-ta-f5151-default-rtdb.asia-southeast1.firebasedatabase.app",
    'projectId': "ta-ta-f5151",
    'storageBucket': "ta-ta-f5151.appspot.com",
    'messagingSenderId': "926896750167",
    'appId': "1:926896750167:web:ee89952a489def4b4fed12",
    'measurementId': "G-SVBSQQJC5D"
}
classes = ["Seledri", "Bawang Merah", "Terong", "Kubis","Tomat","Cengek"]

print("Train dataset file : data-baru.csv")
dataset = pd.read_csv("data-baru.csv")
   #"data-baru.csv", index_col="Tanggal", parse_dates=["Tanggal"]
#)
#
#print(dataset.head())
print("------------------------------------------------------")
time.sleep(1)
#print(dataset.describe())
print("------------------------------------------------------")

training_data = dataset.iloc[:, 7].values
#Apply feature scaling to the data set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_data.reshape(-1, 1))
#print(training_data)
#Initialize our x_training_data and y_training_data variables 
#as empty Python lists
x_training_data = []
y_training_data =[]
for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])
    y_training_data.append(training_data[i, 0])

#Transforming our lists into NumPy arrays
x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)

#print(x_training_data.shape)
#print(y_training_data.shape)

#Reshaping the NumPy array to meet TensorFlow standards
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], 
                                               x_training_data.shape[1], 
                                               1))

#Printing the new shape of x_training_data
print(x_training_data.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

rnn = Sequential()

#Adding our first LSTM layer
rnn.add(LSTM(units = 45, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))

#Perform some dropout regularization
rnn.add(Dropout(0.2))

#Adding three more LSTM layers with dropout regularization
for i in [True, True, False]:
    rnn.add(LSTM(units = 45, return_sequences = i))
    rnn.add(Dropout(0.2))

#Adding our output layer
rnn.add(Dense(units = 1))

#Compiling the recurrent neural network
rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')

#Training the recurrent neural network
rnn.fit(x_training_data, y_training_data, epochs = 10, batch_size = 32)

#Import the test data set and transform it into a NumPy array
test_data = pd.read_csv('test.csv')
test_data = test_data.iloc[:, 7].values

#Make sure the test data's shape makes sense
print(test_data.shape)

#Plot the test data
plt.plot(test_data)

#Create unscaled training data and test data objects
unscaled_training_data = pd.read_csv('data-baru.csv')
unscaled_test_data = pd.read_csv('test.csv')

#Concatenate the unscaled data
all_data = pd.concat((unscaled_training_data['class'], unscaled_test_data['class']), axis = 0)

#Create our x_test_data object, which has each January day + the 40 prior days
x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
x_test_data = np.reshape(x_test_data, (-1, 1))

#Scale the test data
x_test_data = scaler.transform(x_test_data)

#Grouping our test data
final_x_test_data = []
for i in range(40, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-40:i, 0])
final_x_test_data = np.array(final_x_test_data)

#Reshaping the NumPy array to meet TensorFlow standards
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], 
                                               final_x_test_data.shape[1], 
                                               1))

#Generating our predicted values
predictions = rnn.predict(final_x_test_data)

#Plotting our predicted values
plt.clf() #This clears the old plot from our canvas
plt.plot(predictions)

#Unscaling the predicted values and re-plotting the data
unscaled_predictions = scaler.inverse_transform(predictions)
plt.clf() #This clears the first prediction plot from our canvas
plt.plot(unscaled_predictions)

#Plotting the predicted values against Facebook's actual stock price
plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
plt.plot(test_data, color = 'black', label = "Real Data")
plt.title('Facebook Stock Price Predictions')


with open('data-baru.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    #print("------------------------------------------------------")
    #print("Nitrogen \t| Phosphorus \t| Potassium \t| Kelembaban \t| Suhu \t| PH \t| Label")
    #print("------------------------------------------------------")
    values = range(4)
    for row in csv_reader:
        #print(" " + str(line_count+1) + "\t" + row[0]+ "\t" + row[1]+ "\t" + row[2]+ "\t" + row[3]+ "\t" + row[4]+ "\t" + row[5]+ "\t" + row[6])
        label = row[6]
        index = 0
        for  i in values:
            if classes[i]==label :
                index = i
                break
        #print(label + "\tclass : " + str(index))
        line_count += 1
    #print(f'Processed {line_count} lines.')

def prediction(data):
    print("Prediksi tanah = ")
    print(data[0])

def stream_handler(data):
    Kelembaban = data["data"]["Kelembaban"]
    Nitrogen = data["data"]["Nitrogen"]
    Phosphorus = data["data"]["Phosphorus"]
    Potassium = data["data"]["Potassium"]
    Suhu = data["data"]["Suhu"]
    PH = data["data"] ["pH"]
    print("Kelembaban : " + str(Kelembaban))
    print("Nitrogen : " + str(Nitrogen))
    print("Phosphorus : " + str(Phosphorus))
    print("Potassium : " + str(Potassium))
    print("Suhu : " + str(Suhu))
    print("PH : " + str(PH))
    prediction([Kelembaban,Nitrogen,Phosphorus,Potassium,Suhu,PH])    

firebase = pyrebase.initialize_app(config)
db = firebase.database()
db.child("/kesuburan").stream(stream_handler)


