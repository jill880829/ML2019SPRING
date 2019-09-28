import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import backend as K
K.set_learning_phase(1)
import csv
import sys
from keras.models import Model

def read_train_data():
    train_dat = np.genfromtxt(sys.argv[1],dtype = str, delimiter=',')
    train_dat = train_dat[1:15,]
    y_train_num = train_dat[:,0].astype('int')
    x_train_string = train_dat[:,1]
    x_train = np.zeros((len(y_train_num),48*48))
    y_train = np.zeros((len(y_train_num),7))
    for i in range(len(x_train)):
        x_train[i] = np.fromstring(x_train_string[i], dtype=int, sep=' ')
        y_train[i,y_train_num[i]] =1
    x_train = x_train.reshape(-1,48,48,1)
    return x_train,y_train

model = load_model('keras_model1.h5')
xx , yy = read_train_data()

layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(xx[10].reshape(1,48,48,1))

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    #x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x
layer_dict = dict([(layer.name, layer) for layer in model.layers])
def vis_img_in_filter(img = np.array(xx[10]).reshape((1, 48, 48, 1)).astype(np.float64), 
                      layer_name = 'conv2d_2'):
    layer_output = layer_dict[layer_name].output
    img_ascs = list()
    for filter_index in range(layer_output.shape[3]):
        # build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, :, :, filter_index])

        # compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, model.input)[0]

        # normalization trick: we normalize the gradient
        grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

        # this function returns the loss and grads given the input picture
        iterate = K.function([model.input], [loss, grads])

        # step size for gradient ascent
        step = 5.

        img_asc = np.array(img)
        # run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([img_asc])
            img_asc += grads_value * step

        img_asc = img_asc[0]
        img_ascs.append(deprocess_image(img_asc).reshape((48, 48)))
        
    if layer_output.shape[3] >= 35:
        plot_x, plot_y = 6, 6
    elif layer_output.shape[3] >= 23:
        plot_x, plot_y = 4, 6
    elif layer_output.shape[3] >= 11:
        plot_x, plot_y = 2, 6
    else:
        plot_x, plot_y = 1, 2
    fig, ax = plt.subplots(plot_x, plot_y, figsize = (12, 12))
    ax[0, 0].imshow(img.reshape((48, 48)), cmap = 'gray')
    ax[0, 0].set_title('Input image')
    fig.suptitle('Input image and %s filters' % (layer_name,))
    fig.tight_layout(pad = 0.3, rect = [0, 0, 0.9, 0.9])
    for (x, y) in [(i, j) for i in range(plot_x) for j in range(plot_y)]:
        if x == 0 and y == 0:
            continue
        ax[x, y].imshow(img_ascs[x * plot_y + y - 1], cmap = 'gray')
        ax[x, y].set_title('filter %d' % (x * plot_y + y - 1))
    plt.tight_layout()
    plt.savefig(sys.argv[2]+'fig2_1.jpg',cmp='gray')
    plt.close()


def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.tight_layout()
    plt.savefig(sys.argv[2]+'fig2_2.jpg',cmp='gray')

vis_img_in_filter()
display_activation(activations, 6, 6, 1)