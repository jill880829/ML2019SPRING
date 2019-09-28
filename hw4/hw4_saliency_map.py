import numpy as np
import keras
import keras.backend as K
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros
from keras.models import load_model
import matplotlib.pyplot as plt
import csv
import sys

def read_train_data():
    train_dat = np.genfromtxt(sys.argv[1],dtype = str, delimiter=',')
    train_dat = train_dat[1:,]
    y_train_num = train_dat[:,0].astype('int')
    x_train_string = train_dat[:,1]
    x_train = np.zeros((len(y_train_num),48*48))
    y_train = np.zeros((len(y_train_num),7))
    X_train = np.zeros((7,48*48))
    Y_train = np.zeros((7))
    for i in range(len(x_train)):
        x_train[i] = np.fromstring(x_train_string[i], dtype=int, sep=' ')
        Y_train[y_train_num[i]] =y_train_num[i]
        X_train[y_train_num[i]] =x_train[i]
    X_train = X_train.reshape(-1,48,48,1)
    X_train = X_train/255
    return X_train,Y_train

model = load_model('keras_model1.h5')
xx , yy = read_train_data()

class SaliencyMask(object):
    def __init__(self, model, output_index=0):
        pass

    def get_mask(self, input_image):
        pass

    def get_smoothed_mask(self, input_image, stdev_spread=.2, nsamples=50):
        stdev = stdev_spread * (np.max(input_image) - np.min(input_image))

        total_gradients = np.zeros_like(input_image, dtype = np.float64)
        for i in range(nsamples):
            noise = np.random.normal(0, stdev, input_image.shape)
            x_value_plus_noise = input_image + noise

            total_gradients += self.get_mask(x_value_plus_noise)

        return total_gradients / nsamples

class GradientSaliency(SaliencyMask):

    def __init__(self, model, output_index = 0):
        # Define the function to compute the gradient
        input_tensors = [model.input]
        gradients = model.optimizer.get_gradients(model.output[0][output_index], model.input)
        self.compute_gradients = K.function(inputs = input_tensors, outputs = gradients)

    def get_mask(self, input_image):
        # Execute the function to compute the gradient
        x_value = np.expand_dims(input_image, axis=0)
        gradients = self.compute_gradients([x_value])[0][0]

        return gradients

# https://github.com/experiencor/deep-viz-keras/blob/master/visual_backprop.py
class VisualBackprop(SaliencyMask):
    def __init__(self, model, output_index = 0):
        inps = [model.input]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model

    def get_mask(self, input_image):
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(len(self.model.layers) - 1, -1, -1):
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis = 3, keepdims = True)
                layer = layer - np.min(layer)
                layer = layer / (np.max(layer) - np.min(layer) + 1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        return visual_bpr[0]
    
    def _deconv(self, feature_map):
        x = Input(shape = (None, None, 1))
        y = Conv2DTranspose(filters = 1, 
                            kernel_size = (3, 3), 
                            strides = (2, 2), 
                            padding = 'same', 
                            kernel_initializer = Ones(), 
                            bias_initializer = Zeros())(x)

        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]

for i in range(7):
    fig, ax = plt.subplots(1, 5, figsize = (12, 16))
    fig.suptitle('vanilla gradient')
    img = np.array(xx[i])

    vanilla = GradientSaliency(model, i)
    mask = vanilla.get_mask(img)
    filter_mask = (mask > 0.0).reshape((48, 48))
    smooth_mask = vanilla.get_smoothed_mask(img)
    filter_smoothed_mask = (smooth_mask > 0.0).reshape((48, 48))

    ax[0].imshow(img.reshape((48, 48)), cmap = 'gray')
    cax = ax[1].imshow(mask.reshape((48, 48)), cmap = 'jet')
    fig.colorbar(cax, ax = ax[1])
    ax[2].imshow(mask.reshape((48, 48)) * filter_mask, cmap = 'gray')
    cax = ax[3].imshow(mask.reshape((48, 48)), cmap = 'jet')
    fig.colorbar(cax, ax = ax[3])
    ax[4].imshow(smooth_mask.reshape((48, 48)) * filter_smoothed_mask, cmap = 'gray')

    plt.tight_layout()
    plt.savefig(sys.argv[2]+'fig1_'+str(i)+'.jpg',cmp='gray')
    plt.close()