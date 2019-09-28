import numpy as np
import keras
from keras.models import Sequential
from keras.layers import BatchNormalization,Dropout,Dense,Activation,Flatten, Conv2D, MaxPooling2D, Activation, GaussianNoise
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import csv
import sys

def read_train_data():
	train_dat = np.genfromtxt(sys.argv[1],dtype = str, delimiter=',')
	train_dat = train_dat[1:,]
	p = np.random.permutation(train_dat.shape[0])
	train_dat = train_dat[p]
	y_train_num = train_dat[:,0].astype('int')
	x_train_string = train_dat[:,1]
	x_train = np.zeros((len(y_train_num),48*48))
	y_train = np.zeros((len(y_train_num),7))
	for i in range(len(x_train)):
		x_train[i] = np.fromstring(x_train_string[i], dtype=int, sep=' ')
		y_train[i,y_train_num[i]] =1
	x_train = x_train.reshape(-1,48,48,1)
	x_train = (x_train - x_train.mean()) / x_train.std()
	return x_train,y_train

def build_model1():
	model = Sequential()
	model.add(Conv2D(filters = 64, kernel_size = (5,5),input_shape=(48,48,1),activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 64, kernel_size = (5,5),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))

	model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(Conv2D(filters = 128, kernel_size = (3,3),activation = 'relu',padding = 'same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size = (2,2)))
	model.add(Dropout(0.25))
	model.add(Flatten())

	model.add(Dense(units = 512,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(units = 512,activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(units=7,activation='softmax'))
	model.summary()
	return model

def build_model2():
	model = Sequential()
	model.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', input_shape=(48,48,1), activation='selu',kernel_initializer='random_normal'))
	model.add(GaussianNoise(0.05))
	model.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='selu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.15))
	model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='selu'))
	model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='selu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.15))
	model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='selu'))
	model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='selu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.15))
	model.add(Flatten())
	model.add(Dense(units=512,kernel_regularizer=regularizers.l2(0.01)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(Dropout(0.2))
	model.add(Dense(units=256,kernel_regularizer=regularizers.l2(0.01)))
	model.add(BatchNormalization())
	model.add(LeakyReLU())
	model.add(Dropout(0.2))
	model.add(Dense(units=7,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
	model.summary()
	return model

def build_model3():

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=(48, 48, 1),activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(256, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())

    model.add(Conv2D(512, (3, 3), padding='same',activation = 'relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(units=1024, kernel_regularizer=regularizers.l2(0.01),activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.45))

    model.add(Dense(units=1024, kernel_regularizer=regularizers.l2(0.01),activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.45))

    model.add(Dense(units=7, activation='softmax', kernel_regularizer=regularizers.l2(0.01)))
    return model

def train(x_train,y_train):
	model1 = build_model1()
	model2 = build_model2()
	model3 = build_model3()
	opt = Adam(lr = 5e-4)
	model1.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics=['accuracy'])
	model2.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics=['accuracy'])
	model3.compile(loss = 'categorical_crossentropy',optimizer = opt,metrics=['accuracy'])
	val_split = 0.1
	val_cnt = int(len(x_train)*val_split)
	val_x = x_train[:val_cnt, :]
	val_y = y_train[:val_cnt, :]
	train_x = x_train[val_cnt:, :]
	train_y = y_train[val_cnt:, :]

	datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
	                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
	datagen.fit(train_x)
	batch_size = 256
	epochs = 1500
	model1.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
	                    epochs=epochs,steps_per_epoch=len(x_train) / batch_size, verbose=2, validation_data=(val_x, val_y))
	model2.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
	                    epochs=epochs,steps_per_epoch=len(x_train) / batch_size, verbose=2, validation_data=(val_x, val_y))
	model3.fit_generator(datagen.flow(train_x, train_y, batch_size=batch_size),
	                    epochs=epochs,steps_per_epoch=len(x_train) / batch_size, verbose=2, validation_data=(val_x, val_y))
	model1.save('keras_model1.h5')
	model2.save('keras_model2.h5')
	model3.save('keras_model3.h5')

x_train,y_train = read_train_data()
train(x_train,y_train)
