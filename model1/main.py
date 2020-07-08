from PIL import Image
import time
import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical

def loadTrainData():
    x_train = []
    y_train = []
    names = os.listdir(r"C:\Users\louis\Documents\STL-10\train_images")[0:18]
    for name in names:
        img = Image.open("C:\\Users\\louis\\Documents\\STL-10\\train_images\\"+name)
        data = np.asarray(img, dtype = "int32")
        x_train.append(data)
        y_train.append(0)

    for name in os.listdir(r"C:\Users\louis\Documents\STL-10\Glasses"):
        img = Image.open("C:\\Users\\louis\\Documents\\STL-10\\Glasses\\"+name)
        img = img.resize((96,96))
        data = np.asarray(img, dtype = "int32")
        x_train.append(data)
        y_train.append(1)


    x = np.reshape(x_train,(len(x_train),96,96,3))
    y = to_categorical(y_train)

    return x, y

#plt.imshow(x_train[205], cmap='hsv')
#plt.show()
def createModel():
    model = Sequential()
    model.add(Conv2D(filters = 48, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu', input_shape = (96,96,3)))
    model.add(Conv2D(filters = 48, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 96, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(Conv2D(filters = 96, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters = 192, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(Conv2D(filters = 192, kernel_size = (3,3), strides=(1,1), data_format="channels_last", padding = 'same',activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2), padding = 'same', strides = 2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2,activation = 'softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

def trainModel(model,name):
    history = model.fit(x, y, epochs=50, batch_size=18,verbose=1)
    results = model.evaluate(x,y,verbose = 1)
    model.save(name+".h5")
    return model

def makePrediction(model,data):
    if data.ndim == 3:
        classes = model.predict(np.expand_dims(data,axis = 0))
    else:
        classes = model.predict(data)
    return classes

def loadTestData():
    x_test = []
    y_test = []
    names = os.listdir(r"C:\Users\louis\Documents\STL-10\test_images")[0:3]
    for name in names:
        img = Image.open("C:\\Users\\louis\\Documents\\STL-10\\test_images\\"+name)
        data = np.asarray(img, dtype = "int32")
        x_test.append(data)
        y_test.append(0)

    for name in os.listdir(r"C:\Users\louis\Documents\STL-10\Glasses_cv"):
        img = Image.open("C:\\Users\\louis\\Documents\\STL-10\\Glasses_cv\\"+name)
        img = img.resize((96,96))
        data = np.asarray(img, dtype = "int32")
        x_test.append(data)
        y_test.append(1)

    x = np.reshape(x_test,(len(x_test),96,96,3))
    y = to_categorical(y_test)

    return x, y

#model = createModel()
#model = trainModel(model,"cnn4")

model = load_model("cnn4.h5")

img = Image.open("C:\\Users\\louis\\Documents\\STL-10\\long shot\\20200708_000552.jpg")
width, height = img.size
length = int(img.size[0]/8)
step = int(length/4)
l = length/2 #because square


max = 0
maxi, maxj = 0, 0

i = 0
while i*step + length <= width:
    j = 0
    while j*step + length <= height:
        temp = img.crop((i*step, j*step,i*step + length, j*step + length))
        temp = temp.resize((96,96))
        data = np.asarray(temp, dtype = "int32")
        classes = makePrediction(model,data)
        if classes[0][0] > max:
            plt.imshow(data, cmap='hsv')
            plt.pause(1)
            print(i,j)
            max = classes[0][0]
            maxi, maxj = i, j


        j += 1
    i += 1

temp = img.crop((maxi*step, maxj*step,maxi*step + length, maxj*step + length))
temp = temp.resize((96,96))
data = np.asarray(temp, dtype = "int32")
plt.imshow(data, cmap='hsv')
plt.show()
