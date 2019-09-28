import sys
import numpy as np
from PIL import Image
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dropout, Dense, Conv2D, Reshape,Flatten, MaxPooling2D, UpSampling2D , Activation,Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import initializers, regularizers, constraints
from keras.models import load_model
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def read_image():
    image = []
    for i in range(40000):
        img_path = sys.argv[1]+'/%06d.jpg'%(i+1)
        img = Image.open(img_path)
        image.append(np.asarray(img))
        img.close()
    image = np.asarray(image)
    print(image.dtype)
    image = image.astype(float)
    image /= 255
    return image

def save_image(data):
    for i,w in enumerate(data):
        img_path = sys.argv[2]+'/%06d.jpg'%(i+1)
        img = np.clip(w, 0, 1)
        img = Image.fromarray(np.uint8(img*255))
        img.save(img_path)

def build_model():
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(32, (3, 3), padding='same',activation = 'selu')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same',activation='selu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), padding='same',activation='selu')(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    encoded = Dense(1024,activation='relu')(x)

    x = Dense(2048,activation='relu')(encoded)
    x = Reshape((16,16,8))(x)
    x = Conv2D(8, (3, 3), padding='same',activation='selu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), padding='same',activation='selu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), padding='same',activation='selu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(3, (3, 3), padding='same')(x)
    decoded = Activation('sigmoid')(x)

    model = Model(input_img, decoded)
    encoder = Model(input_img, encoded)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.summary()
    return model,encoder

img = read_image()
noise_factor = 0.1
img_noisy = img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=img.shape) 
img_noisy = np.clip(img_noisy,0,1)
model ,encoder = build_model()
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1,
                             shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
datagen.fit(img_noisy)
model.fit_generator(datagen.flow(img_noisy, img, batch_size=256),
                        epochs=100,steps_per_epoch=len(img) / 256, verbose=2)
# model.fit(img_noisy, img, batch_size=256, epochs=500)
model.save('autoencoder.h5')
encoder.save('encoder.h5')
# score = model.evaluate(img, img, verbose=1)
# print(score)
# test = model.predict(img[:100])
# save_image(test)