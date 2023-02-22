import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential, load_model
import pyrebase

import tensorflow as tf
# print(tf._version_)
Kelembaban = 0
Nitrogen = 0
Phosphorus = 0
Potassium = 0
Suhu = 0
PH = 0

print("KLASIFIKASI TANAMAN MENGGUNAKAN RNN")
print("===================================")
print("Konfigurasi firebase")
config = {
    'apiKey': "5FFgTX0nW9Ix0QHtbBVqAZtzpgIeUufuVt1rSCN0",
    'authDomain': "unsurtanah-7f351.firebaseapp.com",
    'databaseURL': "https://unsurtanah-7f351-default-rtdb.firebaseio.com/",
    'projectId': "unsurtanah-7f351",
    'storageBucket': "unsurtanah-7f351.appspot.com",
    'messagingSenderId': "926896750167",
    'appId': "1:926896750167:web:ee89952a489def4b4fed12",
    'measurementId': "G-SVBSQQJC5D"
}
classes = ["Kentang","Cabe","Selada","Bawang Merah","Kembang Kol"]
num_input = 6

dataset = pd.read_csv('data fix.csv', index_col=['date'])
print("Shape of the Dataset:", dataset.shape)
dataset.head()
# preparing input features
feature = dataset.drop(['label','class'], axis=1)
label=dataset['class']
# feature = dataset[['Suhu','Kelembaban']]#,'class']]
feature = dataset[['N','P','K','Kelembaban','Suhu','pH']]#,'class']]
print(len(label))

# x,y = feature.to_numpy(), label.to_numpy()
x, y = feature.values, label.values
print(x.shape, y.shape)

# scaling values for model
x_scale = MinMaxScaler(feature_range=(0, 1))
# y_scale = MinMaxScaler()

X = x_scale.fit_transform(x)
# Y = y_scale.fit_transform(y.reshape(-1,1))

from sklearn.model_selection import StratifiedKFold

k = 4 # number of folds
kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
batch_size = 64

# creating model using Keras
from tensorflow.keras.optimizers import Adam,RMSprop,SGD
# model_gru = Sequential()
model_gru = tf.keras.models.Sequential()
# membuat model gru 
model_gru.add(tf.keras.layers.GRU(units=256, return_sequences=True,input_shape=(num_input,1), activation='relu'))
model_gru.add(tf.keras.layers.GRU(128, activation='relu'))
# model_gru.add(tf.keras.layers.Dropout(0.4))

#mengconeksikan model gru dengan koneksi penuh
model_gru.add(tf.keras.layers.Dense(128, activation='relu'))
model_gru.add(tf.keras.layers.Dense(64, activation='relu'))
# model_gru.add(tf.keras.layers.Dropout(0.4))

# untuk output 5 kelas tanaman
model_gru.add(tf.keras.layers.Dense(5, activation='softmax'))

adam = Adam(learning_rate=0.001)
# RMSprop = RMSprop(learning_rate=0.001)
# SGD = SGD(learning_rate=0.001)


print(model_gru.summary())


weights_list = []
history_list = []
scores = []
# #------ K-Fold Cross validation------
model_gru.compile(optimizer=adam, loss="sparse_categorical_crossentropy", metrics=['accuracy'])
'''
epochs = 40

for i, (train_index, val_index) in enumerate (kfold.split(X,y)):

    X_train, X_val = X[train_index], X[val_index]  
    y_train, y_val = y[train_index], y[val_index]

    print("Data shape for fold ", i)
    print("X_train : ", X_train.shape)
    print("X_val : ", X_val.shape)
    print("y_train : ", y_train.shape)
    print("y_val : ", y_val.shape)
    print("train k-fold", i, "berjalan")

    for k in range(0, 1):
        #------ K-Fold Cross validation------
        # Mereset model sebelum melatih
        # model_gru.reset_states() 
        history_model = model_gru.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_val, y_val))
        model_gru.save("{}.h5".format('model'))
        weights = model_gru.get_weights()
        weights_list.append(weights)
        # Menyimpan data history ke dalam list
        history_list.append(history_model.history)
        scores.append(history_model.history['val_accuracy'][-1])
        mean_loss = np.mean([history['loss'][-1] for history in history_list], axis=0)
        mean_acc = np.mean([history['accuracy'][-1] for history in history_list], axis=0)
        mean_val_loss = np.mean([history['val_loss'][-1] for history in history_list], axis=0)
        mean_val_acc = np.mean([history['val_accuracy'][-1] for history in history_list], axis=0)


        print("Fold ", i, " score : ", history_model.history['val_accuracy'][-1])
        print("Epoch ", epochs, " training berhasil!")
        print("Fold ", i, " training berhasil!")
        print("=================", i, "===============================")
        best_weights = model_gru.get_weights()

# Mencari nilai terbaik dari k-fold
best_score = max(scores)
print("Nilai terbaik dari k-fold adalah : ", best_score)

# Mendapatkan index dari nilai terbaik
best_index = scores.index(best_score)

# Menampilkan hasil dari fold yang memiliki nilai terbaik
print("Hasil dari fold ke-", best_index, " adalah yang terbaik dengan nilai akurasi : ", best_score)

best_weights = weights_list[best_index]
model_gru.set_weights(best_weights)
print("Model dengan nilai terbaik sudah digunakan untuk melakukan testing.")
'''

model_gru =load_model("{}.h5".format('model'))

def prediction(data):
    
    #data = [[11,15,38,81,24.5,5.88]]
    data = [data]
    data = np.array(data)
    Xdata = x_scale.transform(data)
    X_val_ = Xdata.reshape((-1,num_input,1))

    yhat = model_gru.predict(X_val_)
    # y_unscaled = y_scale.inverse_transform(yhat)

    #0 0.33 0.66 1.0
    best_class_index = np.argmax(yhat[0])
    print("Prediksi tanah = ")
    print(classes[0] + " \t: " + "{:.2f}".format(yhat[0][0]) + "%")
    print(classes[1] + " \t: " + "{:.2f}".format(yhat[0][1]) + "%")
    print(classes[2] + " \t: " + "{:.2f}".format(yhat[0][2]) + "%")
    print(classes[3] + " \t: " + "{:.2f}".format(yhat[0][3]) + "%")
    print(classes[4] + " \t: " + "{:.2f}".format(yhat[0][4]) + "%")

    print("Kelas terbaik : ", classes[best_class_index])
    
       
    db.child('/hasil').child(classes[0]).set(f"{yhat[0][0]*100:.2f}%")
    db.child('/hasil').child(classes[1]).set(f"{yhat[0][1]*100:.2f}%")
    db.child('/hasil').child(classes[2]).set(f"{yhat[0][2]*100:.2f}%")
    db.child('/hasil').child(classes[3]).set(f"{yhat[0][3]*100:.2f}%")
    db.child('/hasil').child(classes[4]).set(f"{yhat[0][4]*100:.2f}%")

    db.child('/hasil').child('Tanaman yang paling cocok adalah').set(classes[best_class_index])


def stream_handler(data):
    global Nitrogen
    global Phosphorus
    global Potassium
    global Kelembaban
    global PH
    global Suhu
    print(data)
    if data["event"]=="put" :
       #if data["path"]=="/Kelembaban" :
       Nitrogen = data["data"]["Nitrogen"]  
       #elif data["path"]=="/Nitrogen" :
       Phosphorus = data["data"]["Phosphorus"]   
       #elif data["path"]=="/Phosphorus" :
       Potassium = data["data"]["Potassium"] 
       #elif data["path"]=="/Potassium" :
       Kelembaban = data["data"]["Kelembaban"]
       #elif data["path"]=="/Suhu" :
       PH = data["data"]["pH"]    
       #elif data["path"]=="/pH" :
       Suhu = data["data"]["Suhu"]       

    else :     
        Nitrogen = data["data"]["Nitrogen"]
        Phosphorus = data["data"]["Phosphorus"]
        Potassium = data["data"]["Potassium"]
        Kelembaban = data["data"]["Kelembaban"]
        PH = data["data"] ["pH"]
        Suhu = data["data"]["Suhu"]

    print("Nitrogen : " + str(Nitrogen))
    print("Phosphorus : " + str(Phosphorus))
    print("Potassium : " + str(Potassium))
    print("Kelembaban : " + str(Kelembaban))
    print("PH : " + str(PH))
    print("Suhu : " + str(Suhu))
    #['N','P','K','Kelembaban','Suhu','pH','class']
    prediction([Nitrogen,Phosphorus,Potassium,Kelembaban,Suhu,PH])    #hapus Nitrogen,Phosphorus,Potassium,

firebase = pyrebase.initialize_app(config)
db = firebase.database()
db.child("/kesuburan").stream(stream_handler)