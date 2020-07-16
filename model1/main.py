from PIL import Image
import time
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from keras.models import Sequential, model_from_json, load_model
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras import activations

def loadTrainData():
    x_train = []
    y_train = []
    names = os.listdir(r"C:\Users\louis\Documents\GlassesData\training_data\negative\train_images")[0:100]
    for name in names:
        img = Image.open("C:\\Users\\louis\\Documents\\GlassesData\\training_data\\negative\\train_images\\"+name)
        data = np.asarray(img, dtype = "int32")
        x_train.append(data)
        y_train.append(0)


    for name in os.listdir(r"C:\Users\louis\Documents\GlassesData\training_data\positive"):
        img = Image.open("C:\\Users\\louis\\Documents\\GlassesData\\training_data\\positive\\"+name)
        img = img.resize((96,96))
        data = np.asarray(img, dtype = "int32")
        x_train.append(data)
        y_train.append(1)


    x = np.reshape(x_train,(len(x_train),96,96,3))
    y = to_categorical(y_train)

    return x, y

def makePrediction(model,data):
    if data.ndim == 3:
        classes = model.predict(np.expand_dims(data,axis = 0))
    else:
        classes = model.predict(data)
    return classes

def loadTestData():
    x_test = []
    y_test = []
    names = os.listdir(r"C:\Users\louis\Documents\GlassesData\test_data\negative")[0:10]
    for name in names:
        img = Image.open("C:\\Users\\louis\\Documents\\GlassesData\\test_data\\negative\\"+name)
        data = np.asarray(img, dtype = "int32")
        x_test.append(data)
        y_test.append(0)

    for name in os.listdir(r"C:\Users\louis\Documents\GlassesData\test_data\positive"):
        img = Image.open("C:\\Users\\louis\\Documents\\GlassesData\\test_data\\positive"+name)
        img = img.resize((96,96))
        data = np.asarray(img, dtype = "int32")
        x_test.append(data)
        y_test.append(1)

    x = np.reshape(x_test,(len(x_test),96,96,3))
    y = to_categorical(y_test)

    return x, y

def regionProposal():
    img = Image.open("C:\\Users\\louis\\Documents\\GlassesData\\long shot\\20200708_000552.jpg")
    width, height = img.size
    length = int(img.size[0]/5)
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
            if classes[0][1] > max:
                plt.imshow(data, cmap='hsv')
                plt.pause(0.5)
                print(i,j,classes[0])
                max = classes[0][1]
                maxi, maxj = i, j
            j += 1
        i += 1

    temp = img.crop((maxi*step, maxj*step,maxi*step + length, maxj*step + length))
    temp = temp.resize((96,96))
    data = np.asarray(temp, dtype = "int32")
    plt.imshow(data, cmap='hsv')
    plt.show()

model = Sequential()
model.add(ResNet50(include_top = False, weights = 'imagenet', input_shape = (96,96,3)))
model.add(Dense(2,activation = 'softmax'))
for layer in model.layers[0].layers:
    layer.trainable = False
model.summary()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x, y = loadTrainData()
print(x,y)
history = model.fit(x, y, epochs=10, batch_size=35,verbose=1)
model.save(resnetd2+".h5")

model = trainModel(model,"resnetD2")
x, y = loadTestData()
results = model.evaluate(x,y,batch_size = 3)
print(results)
