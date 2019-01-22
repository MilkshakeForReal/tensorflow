# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:58:56 2019

@author: lenovo
"""

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras import models, layers, optimizers
#from sklearn.model_selection import train_test_split
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data_sparse/", one_hot = True)

x_train, y_train = mnist.train.next_batch(55000)
x_test, y_test = mnist.test.next_batch(10000)
x_val, y_val = mnist.validation.next_batch(5000)

n_train = len(y_train)
n_val = len(y_val)
#set hyperparamters
input_dim = 28
batch_size = 100

def resize(data, dim):
    return np.reshape(data, (data.shape[0], dim, dim, 1))

x_train = resize(x_train, input_dim)
x_val = resize(x_val, input_dim)
x_test = resize(x_test, input_dim)



#IP => [CONV => RELU => BN => POOL]*2 =>[FC => RELU => BN => DO]*2 =>OP
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), input_shape = (input_dim, input_dim,1)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size = (2, 2)))

model.add(layers.Conv2D(64, (3,3)))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPool2D(pool_size = (2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(32))
model.add(layers.Activation('relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(10))
model.add(layers.Activation("softmax"))

model.compile(loss = "categorical_crossentropy",
              optimizer = 'Adam',
              metrics = ['accuracy'])
model.summary()

"""
train_datagen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
"""
history = model.fit(
    x_train, y_train, 
    steps_per_epoch = n_train // batch_size,
    epochs = 40,
    validation_data = (x_val, y_val),
    validation_steps= n_val // batch_size
)

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save_weights('model_wieghts.h5')
model.save('model_keras.h5')

y_pred = model(x_test)
priny((y_pred == y_test).astype(int).sum()/y_pred.shape[0])


