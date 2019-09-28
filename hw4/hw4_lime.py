import numpy as np
import keras
from keras.models import load_model
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.segmentation import mark_boundaries,slic
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import sys
from keras.models import Model
import os,sys
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image


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
images , y = read_train_data()
def predict(img):
    #img = K.eval(image)
    img_rv = img[:,:,:,0]
    img_rv = img_rv.reshape(-1,48,48,1)
    res = model.predict(img_rv)
    print(res.shape)
    return res
def segmentation(image):
    return slic(image,n_segments=300)
def explain(instance, predict,num,segmentation):
    np.random.seed(16)
    return explainer.explain_instance(instance, classifier_fn=predict,top_labels = None,labels=(num, ),segmentation_fn=segmentation)

for i in range(7):
    images_rgb = np.zeros((48,48,3))
    images_rgb[:,:,0]=images[i,:,:,0]
    images_rgb[:,:,1]=images[i,:,:,0]
    images_rgb[:,:,2]=images[i,:,:,0]
    print(images_rgb.shape)
    print(i)
    explainer = lime_image.LimeImageExplainer()
    explaination = explain(images_rgb,predict,i,segmentation)
    image, mask = explaination.get_image_and_mask(
                                    label=i,
                                    positive_only=False,
                                    hide_rest=False,
                                    num_features=5,
                                    min_weight=0.0
                                )

    plt.imsave(sys.argv[2]+'fig3_'+str(i)+'.jpg', image)

# explainer = lime_image.LimeImageExplainer()
# predict_ = lambda x : np.squeeze(model.predict(x[:, :, :, 0].reshape(-1, 48, 48, 1)))
# for i in range(7):
#     image = [images[i]] * 3
#     image = np.concatenate(image, axis = 2)
#     print(np.squeeze(model.predict(image[:, :, 0].reshape(-1, 48, 48, 1))))
#     explanation = explainer.explain_instance(image, predict_, labels=(i, ), top_labels=None, hide_color=0, num_samples=1000)
#     temp, mask = explanation.get_image_and_mask(i, positive_only=True, num_features=1000, hide_rest=True)
#     plt.imsave('lime' + str(i) + '.jpg', mark_boundaries(temp / 2 + 0.5, mask))
